import colorsys
import io
import logging
import math
import os
import tempfile
from contextlib import nullcontext
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from django.conf import settings
from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.coordinate import Coordinate
from api.models.image import ImageModel
from api.models.mask import MaskCategory, SegmentationMask
from api.models.project import Project
from api.serializers import (
    ImageModelSerializer,
    MaskCategorySerializer,
    ProjectSerializer,
)

try:
    from sam3.model_builder import build_sam3_video_model
except ImportError:  # pragma: no cover - safeguards environments without SAM3 installed
    def build_sam3_video_model(*args, **kwargs):
        raise ImportError("sam3 is not installed; build_sam3_video_model is unavailable.")

device = "cuda" if torch.cuda.is_available() else "cpu"

SAM3_CHECKPOINT_PATH = os.getenv("SAM3_CHECKPOINT_PATH")
SAM3_BPE_PATH = os.getenv("SAM3_BPE_PATH")
SAM3_LOAD_FROM_HF = SAM3_CHECKPOINT_PATH is None

# These are shared across requests to avoid reloading the model
# SAM3 video tracker (used for both video tracking and per-image prompts)
sam3_video_model = None
sam3_video_states = {}
sam3_text_obj_map = {}
sam3_obj_category_map = {}
propagation_progress = {}
log = logging.getLogger(__name__)


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
        masks = SegmentationMask.objects.filter(image__project=project)
        count = masks.count()
        for m in masks:
            if m.mask:
                m.mask.delete(save=False)
        masks.delete()
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


class MaskCategoryViewSet(viewsets.ModelViewSet):
    queryset = MaskCategory.objects.all()
    serializer_class = MaskCategorySerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        return MaskCategory.objects.filter(project__user=self.request.user)

    def perform_create(self, serializer):
        project_id = self.request.data.get("project_id")
        project = get_object_or_404(Project, id=project_id, user=self.request.user)
        serializer.save(project=project)


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all()
    serializer_class = ImageModelSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    @staticmethod
    def _resolve_category(project: Project, category_id):
        if category_id:
            return get_object_or_404(MaskCategory, id=category_id, project=project)
        # fallback/default category
        category, _ = MaskCategory.objects.get_or_create(
            project=project, name="Default", defaults={"color": "#00c800"}
        )
        return category

    def _ensure_frame_cache_initialized(self, inference_state):
        """
        SAM3 tracker expects a cached entry per frame before propagation/refinement.
        If the cache is empty (e.g., right after reset_state), pre-seed it with
        empty dicts for every frame to avoid assertion errors during propagation.
        """
        cached = inference_state.get("cached_frame_outputs")
        if cached is None:
            return
        for idx in range(inference_state.get("num_frames", 0)):
            cached.setdefault(idx, {})

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

    @action(detail=True, methods=["post"])
    def generate_mask(self, request, pk=None):
        image: ImageModel = self.get_object()
        project = image.project
        data = request.data
        coordinates = data.get("coordinates") or []
        if not coordinates:
            return Response(
                {"error": "No coordinates provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        category = self._resolve_category(project, data.get("category_id"))

        try:
            model = self._get_sam3_video_model()
            inference_state = self._get_video_inference_state(project, image=image)
        except Exception as exc:
            return Response(
                {"error": f"Failed to load SAM3 model: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
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

        frame_idx = self._frame_index_from_name(image.image.name)
        obj_id = data.get("obj_id")
        if obj_id is None:
            # Prefer obj_id from a prior text prompt for the same category
            obj_id = sam3_text_obj_map.get(project.id, {}).get(category.id, category.id)
        try:
            obj_id = int(obj_id)
        except (TypeError, ValueError):
            obj_id = category.id
        self._ensure_frame_cache_initialized(inference_state)

        with self._get_autocast_context():
            inference_state["cached_frame_outputs"].setdefault(frame_idx, {})
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

        masks = outputs.get("out_binary_masks")
        obj_ids = outputs.get("out_obj_ids")

        # Normalize to Python lists
        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
        if isinstance(masks, np.ndarray):
            masks = list(masks)
        if torch.is_tensor(obj_ids):
            obj_ids = obj_ids.detach().cpu().numpy()
        if isinstance(obj_ids, np.ndarray):
            obj_ids = obj_ids.tolist()
        if obj_ids is None:
            obj_ids = []

        if masks is None or len(masks) == 0:
            return Response(
                {"error": "SAM3 did not return any masks"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Select the mask corresponding to this obj_id (fallback to first)
        selected_idx = 0
        try:
            selected_idx = next(
                i for i, oid in enumerate(obj_ids) if int(oid) == int(obj_id)
            )
        except StopIteration:
            selected_idx = 0

        mask_arr = masks[selected_idx]
        if torch.is_tensor(mask_arr):
            mask_arr = mask_arr.detach().cpu().numpy()
        mask_array = (mask_arr > 0).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array, mode="L")
        temp_buffer = io.BytesIO()
        mask_image.save(temp_buffer, format="PNG")
        temp_buffer.seek(0)

        segmentation_mask, _ = SegmentationMask.objects.get_or_create(
            image=image, category=category
        )
        if segmentation_mask.mask:
            segmentation_mask.mask.delete(save=False)
        segmentation_mask.points = coordinates
        segmentation_mask.mask.save(
            f"{self._frame_stem(image.image.name)}.png",
            ContentFile(temp_buffer.read()),
            save=True,
        )
        segmentation_mask.save(update_fields=["points"])
        serializer = ImageModelSerializer(image, context={"request": request})
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def generate_text_mask(self, request, pk=None):
        image: ImageModel = self.get_object()
        project = image.project
        prompt = (request.data.get("prompt") or "").strip()
        max_masks = request.data.get("max_masks") or 1
        threshold = request.data.get("threshold", 0.5)

        if not prompt:
            return Response(
                {"error": "A text prompt is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            max_masks = max(1, int(max_masks))
        except (TypeError, ValueError):
            max_masks = 1
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.5
        threshold = min(1.0, max(0.01, threshold))
        prompt_base = self._sanitize_prompt_base(prompt)

        try:
            model = self._get_sam3_video_model()
            inference_state = self._get_video_inference_state(project, image=image)
        except Exception as exc:
            return Response(
                {"error": f"Failed to load SAM3 model: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        self._ensure_frame_cache_initialized(inference_state)
        frame_idx = self._frame_index_from_name(image.image.name)
        inference_state["cached_frame_outputs"].setdefault(frame_idx, {})

        # Apply user-provided threshold; the tracker uses this value during forward pass
        inference_state["tracker_confidence_threshold"] = threshold
        with self._get_autocast_context():
            _, outputs = model.add_prompt(
                inference_state=inference_state,
                frame_idx=frame_idx,
                text_str=prompt,
            )

        masks = outputs.get("out_binary_masks") if outputs else None
        scores = outputs.get("out_obj_scores") if outputs else None
        obj_ids = outputs.get("out_obj_ids") if outputs else None

        if masks is None or (torch.is_tensor(masks) and masks.numel() == 0):
            return Response(
                {
                    "error": "SAM3 did not return any masks for this prompt. Try lowering the threshold."
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        if torch.is_tensor(masks):
            masks = masks.detach().cpu()
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().tolist()
        if torch.is_tensor(obj_ids):
            obj_ids = obj_ids.detach().cpu().tolist()
        if isinstance(obj_ids, np.ndarray):
            obj_ids = obj_ids.tolist()

        mask_list = list(masks)
        if not mask_list:
            return Response(
                {
                    "error": "SAM3 did not return any masks for this prompt. Try lowering the threshold."
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        # Remove any prior categories created from this prompt base before adding new ones
        MaskCategory.objects.filter(
            project=project, name__startswith=f"{prompt_base}_"
        ).delete()

        score_list = scores if isinstance(scores, list) else [1.0] * len(mask_list)
        ordered_indices = sorted(
            range(len(mask_list)),
            key=lambda idx: float(score_list[idx]) if idx < len(score_list) else 0,
            reverse=True,
        )

        existing_names = set(
            MaskCategory.objects.filter(project=project).values_list("name", flat=True)
        )
        created_categories = []
        proj_cat_to_obj = sam3_text_obj_map.setdefault(project.id, {})
        proj_obj_to_cat = sam3_obj_category_map.setdefault(project.id, {})
        for local_idx, mask_idx in enumerate(ordered_indices[: max_masks or 1]):
            category_name = self._next_category_name(
                base_name=prompt_base, existing_names=existing_names, start_at=local_idx
            )
            existing_names.add(category_name)
            color = self._color_for_index(len(existing_names))
            category = MaskCategory.objects.create(
                project=project, name=category_name, color=color
            )
            created_categories.append(category)

            mask_arr = mask_list[mask_idx]
            if torch.is_tensor(mask_arr):
                mask_arr = mask_arr.detach().cpu().numpy()

            if isinstance(mask_arr, np.ndarray):
                mask_arr = np.squeeze(mask_arr)
                if mask_arr.ndim > 2:
                    # Fallback: take the first channel if squeezing still leaves depth
                    mask_arr = mask_arr[..., 0]
            binary_mask = (mask_arr > 0).astype(np.uint8) * 255
            mask_image = Image.fromarray(binary_mask, mode="L")
            temp_buffer = io.BytesIO()
            mask_image.save(temp_buffer, format="PNG")
            temp_buffer.seek(0)

            segmentation_mask, _ = SegmentationMask.objects.get_or_create(
                image=image, category=category
            )
            if segmentation_mask.mask:
                segmentation_mask.mask.delete(save=False)
            segmentation_mask.points = []
            segmentation_mask.mask.save(
                f"{self._frame_stem(image.image.name)}_{category.id}.png",
                ContentFile(temp_buffer.read()),
                save=True,
            )
            segmentation_mask.save(update_fields=["points"])

            # Track obj_id mapping so future point prompts refine this object
            try:
                obj_id_value = (
                    obj_ids[mask_idx] if obj_ids and mask_idx < len(obj_ids) else None
                )
                if obj_id_value is None:
                    obj_id_value = mask_idx  # fallback so we always have a mapping
                obj_id_int = int(obj_id_value)
                proj_cat_to_obj[category.id] = obj_id_int
                proj_obj_to_cat[obj_id_int] = category.id
            except Exception as exc:
                log.warning(
                    "[text-mask] failed mapping category_id=%s obj_id_value=%s (%s)",
                    category.id,
                    obj_id_value,
                    exc,
                )

        refreshed_image = ImageModel.objects.get(pk=image.pk)
        image_data = ImageModelSerializer(
            refreshed_image, context={"request": request}
        ).data
        category_data = MaskCategorySerializer(
            MaskCategory.objects.filter(project=project), many=True
        ).data

        return Response(
            {
                "image": image_data,
                "categories": category_data,
                "created_categories": MaskCategorySerializer(
                    created_categories, many=True
                ).data,
            }
        )

    @action(detail=False, methods=["post"])
    def propagate_mask(self, request):
        project_id = request.data.get("project_id")
        category_id = request.data.get("category_id")
        if not project_id:
            return Response(
                {"error": "project_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        project_obj = get_object_or_404(Project, id=project_id, user=request.user)
        project_images = ImageModel.objects.filter(project=project_obj)
        if project_obj.type == "segmentation":
            return Response(
                {
                    "error": "Mask propagation is only supported for video tracking projects."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            model = self._get_sam3_video_model()
            inference_state = self._get_video_inference_state(project_obj)
            categories_by_id = {
                c.id: c for c in MaskCategory.objects.filter(project=project_obj)
            }
            # Map SAM3 obj_ids that came from text prompts back to their categories so
            # propagation doesn't collapse multiple tracked objects into the default.
            text_obj_map = sam3_text_obj_map.get(project_obj.id, {})
            for cat_id, obj_id in text_obj_map.items():
                try:
                    cat = categories_by_id.get(int(cat_id))
                    if cat is not None:
                        categories_by_id[int(obj_id)] = cat
                except (TypeError, ValueError):
                    continue
            obj_category_map = sam3_obj_category_map.get(project_obj.id, {})
            for obj_id, cat_id in obj_category_map.items():
                try:
                    cat = categories_by_id.get(int(cat_id))
                    if cat is not None:
                        categories_by_id[int(obj_id)] = cat
                except (TypeError, ValueError):
                    continue
            default_category = categories_by_id.get(
                category_id
            ) or self._resolve_category(project_obj, category_id)
            if default_category.id not in categories_by_id:
                categories_by_id[default_category.id] = default_category
        except Exception as exc:
            return Response(
                {"error": f"Failed to load SAM3 model: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        self._ensure_frame_cache_initialized(inference_state)
        total_frames = project_images.count()
        if total_frames == 0:
            return Response(
                {"error": "No frames available to propagate masks."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        self._set_propagation_progress(project_obj.id, 0, total_frames, "running")
        # SAM3 holds back the first `hotstart_delay` frames before yielding anything, so
        # offset progress to match the model's tqdm bar rather than the yielded frames.
        progress_offset = int(getattr(model, "hotstart_delay", 0))
        processed_frames = 0
        with self._get_autocast_context():
            try:
                for out_frame_idx, outputs in model.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=None,
                    reverse=False,
                ):
                    processed_frames += 1
                    effective_frames = min(
                        processed_frames + progress_offset, total_frames
                    )
                    self._set_propagation_progress(
                        project_obj.id, effective_frames, total_frames, "running"
                    )
                    if outputs is None:
                        log.warning("[propagate] frame=%s outputs=None", out_frame_idx)
                        continue
                    masks = outputs.get("out_binary_masks")
                    obj_ids = outputs.get("out_obj_ids")
                    if masks is None or obj_ids is None:
                        log.warning(
                            "[propagate] frame=%s missing masks/obj_ids (masks=%s obj_ids=%s)",
                            out_frame_idx,
                            type(masks).__name__ if masks is not None else None,
                            type(obj_ids).__name__ if obj_ids is not None else None,
                        )
                        continue

                    # Normalize possible tensor/ndarray outputs into Python lists
                    if torch.is_tensor(masks):
                        masks = masks.detach().cpu().numpy()
                    if isinstance(masks, np.ndarray):
                        masks = list(masks)
                    if torch.is_tensor(obj_ids):
                        obj_ids = obj_ids.detach().cpu().numpy()
                    if isinstance(obj_ids, np.ndarray):
                        obj_ids = obj_ids.tolist()

                    matching_image = project_images.filter(
                        image__endswith=f"{out_frame_idx:05d}.jpg"
                    ).first()
                    if not matching_image or len(masks) == 0:
                        log.warning(
                            "[propagate] frame=%s no matching image (%s) or masks empty",
                            out_frame_idx,
                            bool(matching_image),
                        )
                        continue

                    # SAM3 returns one mask per tracked object; persist them all
                    for idx, obj_id in enumerate(obj_ids):
                        if idx >= len(masks):
                            continue
                        category = categories_by_id.get(int(obj_id), default_category)
                        mask_arr = masks[idx]
                        if torch.is_tensor(mask_arr):
                            mask_arr = mask_arr.detach().cpu().numpy()
                        binary_mask = (mask_arr > 0).astype(np.uint8) * 255
                        mask_image = Image.fromarray(binary_mask, mode="L")
                        temp_buffer = io.BytesIO()
                        mask_image.save(temp_buffer, format="PNG")
                        temp_buffer.seek(0)
                        seg_mask, _ = SegmentationMask.objects.get_or_create(
                            image=matching_image, category=category
                        )
                        if seg_mask.mask:
                            seg_mask.mask.delete(save=False)
                        seg_mask.mask.save(
                            f"{self._frame_stem(matching_image.image.name)}.png",
                            ContentFile(temp_buffer.read()),
                            save=True,
                        )
            except Exception as exc:
                self._set_propagation_progress(
                    project_obj.id, 0, 0, "failed", detail=str(exc)
                )
                log.exception("Propagation failed for project %s", project_obj.id)
                return Response(
                    {"error": f"Failed to propagate masks: {exc}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            finally:
                try:
                    model.reset_state(inference_state)
                except Exception:
                    pass

        self._set_propagation_progress(project_obj.id, 0, 0, "idle")

        serializer = ImageModelSerializer(
            project_images, many=True, context={"request": request}
        )
        return Response(serializer.data)

    @action(detail=False, methods=["get"])
    def unload_model(self, request):
        global sam3_video_model, sam3_video_states, sam3_text_obj_map, sam3_obj_category_map
        sam3_video_states = {}
        sam3_text_obj_map = {}
        sam3_obj_category_map = {}
        # Keep models resident; just clear caches and GPU memory where possible.
        if sam3_video_model is not None:
            try:
                sam3_video_model.reset_state()
            except Exception:
                pass
        if device == "cuda":
            torch.cuda.empty_cache()
        return Response({"message": "SAM3 caches cleared; model kept loaded"})

    @action(detail=True, methods=["delete"])
    def delete_mask(self, request, pk=None):
        image: ImageModel = self.get_object()
        category_id = request.query_params.get("category_id")
        masks_qs = image.masks.all()
        if category_id:
            masks_qs = masks_qs.filter(category_id=category_id)
        deleted = 0
        for m in masks_qs:
            if m.mask:
                m.mask.delete(save=False)
            m.delete()
            deleted += 1
        return Response({"detail": f"Deleted {deleted} mask(s)."})

    @action(detail=True, methods=["delete"])
    def delete_coordinates(self, request, pk=None):
        image: ImageModel = self.get_object()
        image.coordinates.all().delete()
        return Response({"detail": "Coordinates have been deleted."})

    @action(detail=False, methods=["get"])
    def propagation_progress(self, request):
        """
        Returns the current propagation progress for a project so the frontend
        can poll. Always starts at 0 for a new run.
        """
        project_id = request.query_params.get("project_id")
        if not project_id:
            return Response(
                {"error": "project_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        project = get_object_or_404(Project, id=project_id, user=request.user)
        progress_payload = propagation_progress.get(
            project.id,
            {"progress": 0, "current": 0, "total": 0, "status": "idle"},
        )
        return Response(progress_payload)

    @staticmethod
    def _color_for_index(index: int) -> str:
        """
        Generate a visually distinct color. Uses a golden-ratio hue step so
        successive calls produce separated hues and includes a gentle alpha
        for nicer overlays on the frontend.
        """
        hue = (index * 0.61803398875) % 1.0
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.65, 0.95)]
        return f"rgba({r},{g},{b},0.6)"

    @staticmethod
    def _sanitize_prompt_base(name: str) -> str:
        return (name.replace("\n", " ").strip() or "prompt")[:200]

    @staticmethod
    def _next_category_name(
        base_name: str, existing_names: set[str], start_at: int
    ) -> str:
        """
        Generate the next available <prompt>_<id> name while respecting the
        unique constraint on categories.
        """
        idx = start_at
        sanitized = ImageViewSet._sanitize_prompt_base(base_name)
        while True:
            candidate = f"{sanitized}_{idx}"
            if candidate not in existing_names:
                return candidate
            idx += 1

    @staticmethod
    def _get_sam3_video_model():
        global sam3_video_model
        if sam3_video_model is None:
            sam3_video_model = build_sam3_video_model(
                checkpoint_path=SAM3_CHECKPOINT_PATH,
                bpe_path=SAM3_BPE_PATH,
                load_from_HF=SAM3_LOAD_FROM_HF,
                device=device,
            )
        return sam3_video_model

    def _get_video_inference_state(
        self, project: Project, image: Optional[ImageModel] = None
    ):
        """
        For simple segmentation projects, only load the single active image into SAM3
        so switching images resets the model and avoids loading all project images.
        Video tracking projects keep a project-level state for propagation across frames.
        """
        cache_entry = sam3_video_states.get(project.pk)
        # Normalize legacy cache shape (plain inference_state) into a dict
        if cache_entry is not None and not isinstance(cache_entry, dict):
            cache_entry = {"state": cache_entry}

        if project.type == "segmentation":
            if image is None:
                raise ValueError("Image is required for segmentation inference state")
            if (
                not cache_entry
                or cache_entry.get("project_type") != "segmentation"
                or cache_entry.get("image_id") != image.pk
            ):
                inference_state = self._get_sam3_video_model().init_state(
                    resource_path=image.image.path,
                    offload_video_to_cpu=device != "cuda",
                    async_loading_frames=False,
                )
                cache_entry = {
                    "project_type": "segmentation",
                    "image_id": image.pk,
                    "state": inference_state,
                }
                sam3_video_states[project.pk] = cache_entry
        else:
            if cache_entry is None or cache_entry.get("project_type") != project.type:
                project_frames = os.path.join(
                    settings.MEDIA_ROOT, "projects", str(project.pk), "images"
                )
                inference_state = self._get_sam3_video_model().init_state(
                    resource_path=project_frames,
                    offload_video_to_cpu=device != "cuda",
                    async_loading_frames=False,
                )
                cache_entry = {
                    "project_type": project.type,
                    "state": inference_state,
                }
                sam3_video_states[project.pk] = cache_entry

        inference_state = cache_entry["state"]
        self._ensure_frame_cache_initialized(inference_state)
        return inference_state

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

    @staticmethod
    def _set_propagation_progress(
        project_id: int,
        current: int,
        total: int,
        status_str: str,
        detail: Optional[str] = None,
    ) -> None:
        safe_total = max(1, int(total or 0))
        safe_current = min(max(int(current), 0), safe_total)
        percent = int((safe_current / safe_total) * 100)
        propagation_progress[project_id] = {
            "progress": percent,
            "current": safe_current,
            "total": safe_total,
            "status": status_str,
            "detail": detail,
        }


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
            ImageViewSet._get_sam3_video_model()
            return Response({"message": "SAM3 models loaded"})
        except Exception as exc:
            return Response(
                {"error": f"Failed to load SAM3: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def unload_model(self, request):
        return ImageViewSet.unload_model(self, request)
