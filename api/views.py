import io
import os
import tempfile
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image
from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.coordinate import Coordinate
from api.models.image import ImageModel
from api.models.project import Project
from api.serializers import (
    CoordinateSerializer,
    ImageModelSerializer,
    ProjectSerializer,
)
from sam3.model_builder import build_sam3_video_model

device = "cuda" if torch.cuda.is_available() else "cpu"

SAM3_CHECKPOINT_PATH = os.getenv("SAM3_CHECKPOINT_PATH")
SAM3_BPE_PATH = os.getenv("SAM3_BPE_PATH")
SAM3_LOAD_FROM_HF = SAM3_CHECKPOINT_PATH is None

# These are shared across requests to avoid reloading the model
sam3_video_model = None
sam3_inference_states = {}


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        return Project.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["delete"])
    def delete_masks(self, request, pk=None):
        """
        Deletes the mask files for all images associated with this project.
        """
        project = self.get_object()
        project_images = ImageModel.objects.filter(project=project)
        count = 0
        for image in project_images:
            if image.mask:
                image.mask.delete(save=True)
                count += 1
        return Response(
            {"detail": f"Deleted masks for {count} images in the project."},
            status=status.HTTP_200_OK,
        )

    @action(detail=True, methods=["delete"])
    def delete_coordinates(self, request, pk=None):
        """
        Deletes all coordinates associated with images in this project.
        """
        project = self.get_object()
        project_images = ImageModel.objects.filter(project=project)
        total_deleted = 0
        for image in project_images:
            deleted, _ = image.coordinates.all().delete()
            total_deleted += deleted
        return Response(
            {"detail": f"Deleted {total_deleted} coordinate entries from the project."},
            status=status.HTTP_200_OK,
        )


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all()
    serializer_class = ImageModelSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        return ImageModel.objects.filter(project__user=self.request.user)

    def _get_autocast_context(self):
        if device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def create(self, request, *args, **kwargs):
        project_id = request.data.get("project_id")
        is_label = request.data.get("is_label", False)

        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return Response(
                {
                    "error": "Project not found or you do not have permission to access it."
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        images = request.FILES.getlist("images")
        image_records = []

        for image in images:
            image_record = ImageModel.objects.create(
                image=image,
                is_label=is_label,
                project=project,
                original_filename=image.name,
            )
            image_records.append(image_record)

        serializer = self.get_serializer(
            image_records, many=True, context={"request": request}
        )
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=["get"])
    def folder_coordinates(self, request):
        folder_path = request.query_params.get("folder_path")
        if not folder_path:
            return Response(
                {"error": "folder_path query parameter is required"}, status=400
            )

        images = ImageModel.objects.filter(folder_path=folder_path)
        all_coordinates = []
        for image in images:
            coordinates = image.coordinates.all()
            serializer = CoordinateSerializer(coordinates, many=True)
            all_coordinates.extend(serializer.data)

        return Response(all_coordinates)

    @action(detail=False, methods=["post"])
    def save_all_coordinates(self, request):
        all_coordinates = request.data
        if not all_coordinates:
            return Response(
                {"error": "No coordinates provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        with transaction.atomic():
            for item in all_coordinates:
                image_id = item.get("image_id")
                coordinates = item.get("coordinates", [])

                # Validate coordinates exist
                if not coordinates:
                    continue

                # Fetch the image
                try:
                    image = ImageModel.objects.get(id=image_id)
                except ImageModel.DoesNotExist:
                    continue

                # Delete existing coordinates for the image
                image.coordinates.all().delete()

                # Create new coordinate objects
                coordinate_objs = [
                    Coordinate(image=image, x=coord["x"], y=coord["y"])
                    for coord in coordinates
                ]
                Coordinate.objects.bulk_create(coordinate_objs)

        return Response({"status": "success"}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"])
    def generate_mask(self, request, pk=None):
        image: ImageModel = self.get_object()
        project = image.project

        try:
            model = self._get_sam3_model()
            inference_state = self._get_inference_state(project)
        except Exception as exc:
            return Response(
                {"error": f"Failed to load SAM3 model: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        data = request.data
        coordinates = data.get("coordinates") or []
        if not coordinates:
            return Response(
                {"error": "No coordinates provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        width, height = self._get_image_dimensions(image.image.path)
        normalized_points = np.array(
            [[coord["x"] / width, coord["y"] / height] for coord in coordinates],
            dtype=np.float32,
        )
        input_labels = np.array(
            [1 if coord.get("include", True) else 0 for coord in coordinates],
            dtype=np.int32,
        )

        # Persist the clicked points
        image.coordinates.all().delete()
        for coord in coordinates:
            Coordinate.objects.create(
                image=image,
                x=coord["x"],
                y=coord["y"],
                include=coord.get("include", True),
            )

        frame_idx = self._frame_index_from_name(image.image.name)
        obj_id = data.get("obj_id", 0)

        with self._get_autocast_context():
            _, outputs = model.add_prompt(
                inference_state=inference_state,
                frame_idx=frame_idx,
                points=normalized_points.tolist(),
                point_labels=input_labels.tolist(),
                obj_id=obj_id,
            )

        if outputs is None or outputs.get("out_binary_masks") is None:
            return Response(
                {"error": "SAM3 did not return any masks"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        binary_masks = outputs["out_binary_masks"]
        if len(binary_masks) == 0:
            return Response(
                {"error": "SAM3 did not return any masks"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        mask_array = binary_masks[0].astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array, mode="L")
        temp_buffer = io.BytesIO()
        mask_image.save(temp_buffer, format="PNG")
        temp_buffer.seek(0)
        if image.mask:
            image.mask.delete(save=False)
        image.mask.save(
            f"{self._frame_stem(image.image.name)}.png",
            ContentFile(temp_buffer.read()),
            save=True,
        )
        serializer = ImageModelSerializer(image, context={"request": request})
        return Response(serializer.data)

    @action(detail=False, methods=["post"])
    def propagate_mask(self, request):
        project = request.data.get("project_id")
        if not project:
            return Response(
                {"error": "project_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        project_images = ImageModel.objects.filter(project=project)
        video_segments = {}
        try:
            model = self._get_sam3_model()
            inference_state = self._get_inference_state(
                get_object_or_404(Project, id=project)
            )
        except Exception as exc:
            return Response(
                {"error": f"Failed to load SAM3 model: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        with self._get_autocast_context():
            for out_frame_idx, outputs in model.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=None,
                reverse=False,
            ):
                if outputs is None:
                    continue
                masks = outputs.get("out_binary_masks")
                obj_ids = outputs.get("out_obj_ids")
                if masks is None or len(masks) == 0:
                    continue
                video_segments[out_frame_idx] = {
                    obj_ids[i]: masks[i] for i in range(len(masks))
                }
                matching_image = project_images.filter(
                    image__endswith=f"{out_frame_idx:05d}.jpg"
                ).first()
                if not matching_image:
                    continue
                binary_mask = masks[0].astype(np.uint8) * 255
                mask_image = Image.fromarray(binary_mask, mode="L")
                temp_buffer = io.BytesIO()
                mask_image.save(temp_buffer, format="PNG")
                temp_buffer.seek(0)
                if matching_image.mask:
                    matching_image.mask.delete(save=False)
                matching_image.mask.save(
                    f"{self._frame_stem(matching_image.image.name)}.png",
                    ContentFile(temp_buffer.read()),
                    save=True,
                )
            model.reset_state(inference_state)

        serializer = ImageModelSerializer(
            project_images, many=True, context={"request": request}
        )
        return Response(serializer.data)

    @action(detail=False, methods=["get"])
    def unload_model(self, request):
        global sam3_video_model, sam3_inference_states
        if sam3_video_model is not None:
            sam3_video_model = None
            sam3_inference_states = {}
            if device == "cuda":
                torch.cuda.empty_cache()
            return Response({"message": "SAM3 model unloaded"})
        return Response({"message": "Model is not loaded"})

    @action(detail=True, methods=["delete"])
    def delete_mask(self, request, pk=None):
        image: ImageModel = self.get_object()
        if image.mask:
            image.mask.delete(save=True)
        return Response({"detail": "Mask has been deleted."})

    @action(detail=True, methods=["delete"])
    def delete_coordinates(self, request, pk=None):
        image: ImageModel = self.get_object()
        image.coordinates.all().delete()
        return Response({"detail": "Coordinates have been deleted."})

    @staticmethod
    def _get_sam3_model():
        global sam3_video_model
        if sam3_video_model is None:
            sam3_video_model = build_sam3_video_model(
                checkpoint_path=SAM3_CHECKPOINT_PATH,
                bpe_path=SAM3_BPE_PATH,
                load_from_HF=SAM3_LOAD_FROM_HF,
                device=device,
            )
        return sam3_video_model

    def _get_inference_state(self, project: Project):
        if project.pk not in sam3_inference_states:
            project_frames = os.path.join(
                settings.MEDIA_ROOT, "projects", str(project.pk), "images"
            )
            sam3_inference_states[project.pk] = self._get_sam3_model().init_state(
                resource_path=project_frames,
                offload_video_to_cpu=device != "cuda",
                async_loading_frames=False,
            )
        return sam3_inference_states[project.pk]

    @staticmethod
    def _get_image_dimensions(image_path: str):
        with Image.open(image_path) as img:
            return img.size

    @staticmethod
    def _frame_index_from_name(path: str) -> int:
        try:
            return int(ImageViewSet._frame_stem(path))
        except ValueError:
            return 0

    @staticmethod
    def _frame_stem(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]


class VideoViewSet(viewsets.ViewSet):
    """
    Handles video uploads. Extracts frames with configurable stride and maximum frames,
    saves them to ImageVideoModel, and returns the created frame records.
    """

    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def create(self, request, *args, **kwargs):
        project_id = request.data.get("project_id")
        is_label = request.data.get("is_label", False)
        stride = int(request.data.get("stride", 1))
        max_frames = int(request.data.get("max_frames", 500))

        project = get_object_or_404(Project, id=project_id, user=request.user)

        # Expect a single "video" file
        video_file = request.FILES.get("video")
        if not video_file:
            return Response(
                {"error": "No video file found in the request."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not video_file.content_type.startswith("video/"):
            return Response(
                {"error": "Uploaded file is not a video."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 1) Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
            for chunk in video_file.chunks():
                temp.write(chunk)
            temp_name = temp.name

        # 2) Extract frames using OpenCV
        cap = cv2.VideoCapture(temp_name)
        frame_records = []
        frame_index = 0
        saved_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # no more frames

            # Skip frames based on stride
            if frame_index % stride != 0:
                frame_index += 1
                continue

            # Stop if max_frames limit is reached
            if saved_frames >= max_frames:
                break

            # Convert the OpenCV frame to bytes
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                frame_index += 1
                continue

            frame_name = f"{saved_frames:05d}.jpg"
            frame_content = ContentFile(buffer.tobytes(), name=frame_name)

            frame_model = ImageModel.objects.create(
                image=frame_content,
                is_label=is_label,
                project=project,
                original_filename=frame_name,
            )
            frame_records.append(frame_model)
            saved_frames += 1
            frame_index += 1

        cap.release()

        serializer = ImageModelSerializer(
            frame_records, many=True, context={"request": request}
        )
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class ModelManagerViewSet(viewsets.ViewSet):
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    @action(detail=False, methods=["post"])
    def load_model(self, request):
        try:
            ImageViewSet._get_sam3_model()
            return Response({"message": "SAM3 model loaded"})
        except Exception as exc:
            return Response(
                {"error": f"Failed to load SAM3: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def unload_model(self, request):
        return ImageViewSet.unload_model(self, request)
