import os
from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.image import ImageModel
from api.models.mask import MaskCategory, SegmentationMask
from api.models.ocr import OcrAnnotation
from api.models.project import Project
from api.models.snapshot import (
    ProjectSnapshot,
    ProjectSnapshotCategory,
    ProjectSnapshotImage,
    ProjectSnapshotOcrAnnotation,
    ProjectSnapshotSegmentationMask,
)
from api.serializers import ImageModelSerializer, MaskCategorySerializer, ProjectSerializer


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

    def _format_snapshot(self, snapshot: ProjectSnapshot):
        title = (snapshot.name or "").strip()
        created_label = snapshot.created_at.strftime("%Y-%m-%d %H:%M")
        return {
            "id": snapshot.id,
            "created_at": snapshot.created_at.isoformat(),
            "name": title,
            "label": title or created_label,
        }

    def _create_snapshot(self, project: Project, name: str = "") -> ProjectSnapshot:
        snapshot = ProjectSnapshot.objects.create(project=project, name=name or "")
        category_map = {}
        for category in project.categories.all():
            snap_category = ProjectSnapshotCategory.objects.create(
                snapshot=snapshot,
                name=category.name,
                color=category.color,
                original_category=category,
            )
            category_map[category.id] = snap_category

        for image in project.images.all():
            snap_image = ProjectSnapshotImage.objects.create(snapshot=snapshot, image=image)
            if project.type in ("segmentation", "video_tracking_segmentation"):
                for mask in image.masks.all():
                    snap_mask = ProjectSnapshotSegmentationMask(
                        snapshot_image=snap_image,
                        category=category_map.get(mask.category_id),
                        points=mask.points or [],
                    )
                    if mask.mask:
                        mask.mask.open("rb")
                        snap_mask.mask.save(
                            os.path.basename(mask.mask.name),
                            ContentFile(mask.mask.read()),
                            save=False,
                        )
                        mask.mask.close()
                    snap_mask.save()

            if project.type in ("ocr", "ocr_kie"):
                for ann in image.ocr_annotations.all():
                    ProjectSnapshotOcrAnnotation.objects.create(
                        snapshot_image=snap_image,
                        shape_type=ann.shape_type,
                        points=ann.points or [],
                        text=ann.text or "",
                        category=ann.category,
                    )

        return snapshot

    def _ensure_categories(self, project: Project, snapshot: ProjectSnapshot):
        category_map = {}
        for snap_cat in snapshot.categories.all():
            category, created = MaskCategory.objects.get_or_create(
                project=project,
                name=snap_cat.name,
                defaults={"color": snap_cat.color},
            )
            if not created and category.color != snap_cat.color:
                category.color = snap_cat.color
                category.save(update_fields=["color"])
            category_map[snap_cat.id] = category
        return category_map

    def _clear_masks(self, image: ImageModel):
        for mask in image.masks.all():
            if mask.mask:
                mask.mask.delete(save=False)
        image.masks.all().delete()

    def _apply_snapshot(
        self,
        project: Project,
        snapshot: ProjectSnapshot,
        target_image_ids=None,
    ):
        category_map = self._ensure_categories(project, snapshot)
        image_qs = snapshot.images.select_related("image").prefetch_related(
            "segmentation_masks",
            "segmentation_masks__category",
            "ocr_annotations",
        )
        if target_image_ids:
            image_qs = image_qs.filter(image_id__in=target_image_ids)

        for snap_image in image_qs:
            image = snap_image.image
            if not image or image.project_id != project.id:
                continue

            if project.type in ("segmentation", "video_tracking_segmentation"):
                self._clear_masks(image)
                for saved_mask in snap_image.segmentation_masks.all():
                    category = category_map.get(saved_mask.category_id)
                    if category is None:
                        category, _ = MaskCategory.objects.get_or_create(
                            project=project,
                            name=saved_mask.category.name if saved_mask.category else "Restored",
                            defaults={"color": "#00c800"},
                        )
                    seg_mask, _ = SegmentationMask.objects.get_or_create(
                        image=image, category=category
                    )
                    if seg_mask.mask:
                        seg_mask.mask.delete(save=False)
                    if saved_mask.mask:
                        saved_mask.mask.open("rb")
                        filename = os.path.basename(saved_mask.mask.name) or f"{image.id}_{category.id}_mask.png"
                        seg_mask.mask.save(
                            filename,
                            ContentFile(saved_mask.mask.read()),
                            save=False,
                        )
                        saved_mask.mask.close()
                    seg_mask.points = saved_mask.points or []
                    seg_mask.save()

            if project.type in ("ocr", "ocr_kie"):
                image.ocr_annotations.all().delete()
                restored = [
                    OcrAnnotation(
                        image=image,
                        shape_type=ann.shape_type,
                        points=ann.points or [],
                        text=ann.text or "",
                        category=ann.category,
                    )
                    for ann in snap_image.ocr_annotations.all()
                ]
                if restored:
                    OcrAnnotation.objects.bulk_create(restored)

    def _delete_snapshot(self, snapshot: ProjectSnapshot):
        for snap_image in snapshot.images.prefetch_related("segmentation_masks", "ocr_annotations"):
            for saved_mask in snap_image.segmentation_masks.all():
                if saved_mask.mask:
                    saved_mask.mask.delete(save=False)
            snap_image.segmentation_masks.all().delete()
            snap_image.ocr_annotations.all().delete()
        snapshot.delete()

    @action(detail=True, methods=["get", "post"], url_path="snapshots")
    def snapshots(self, request, pk=None):
        project = self.get_object()
        if request.method == "GET":
            snapshots = project.snapshots.order_by("-created_at")
            data = [self._format_snapshot(s) for s in snapshots]
            return Response({"snapshots": data})

        name = (request.data.get("name") or "").strip()
        snapshot = self._create_snapshot(project, name=name)
        return Response({"snapshot": self._format_snapshot(snapshot)}, status=status.HTTP_201_CREATED)

    @action(
        detail=True,
        methods=["post"],
        url_path="snapshots/(?P<snapshot_id>[^/.]+)/load_project",
    )
    def load_project_snapshot(self, request, pk=None, snapshot_id=None):
        project = self.get_object()
        snapshot = get_object_or_404(ProjectSnapshot, id=snapshot_id, project=project)
        self._apply_snapshot(project, snapshot)
        serializer = ProjectSerializer(project, context={"request": request})
        return Response(
            {"project": serializer.data, "snapshot": self._format_snapshot(snapshot)},
            status=status.HTTP_200_OK,
        )

    @action(
        detail=True,
        methods=["post"],
        url_path="snapshots/(?P<snapshot_id>[^/.]+)/load_page",
    )
    def load_page_snapshot(self, request, pk=None, snapshot_id=None):
        project = self.get_object()
        image_id = request.data.get("image_id")
        if not image_id:
            return Response(
                {"detail": "image_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            image_id = int(image_id)
        except (TypeError, ValueError):
            return Response(
                {"detail": "image_id must be an integer."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        snapshot = get_object_or_404(ProjectSnapshot, id=snapshot_id, project=project)
        self._apply_snapshot(project, snapshot, target_image_ids=[image_id])

        image = get_object_or_404(ImageModel, id=image_id, project=project)
        image_data = ImageModelSerializer(image, context={"request": request}).data
        categories = MaskCategorySerializer(
            project.categories.all(), many=True, context={"request": request}
        ).data
        return Response(
            {
                "image": image_data,
                "categories": categories,
                "snapshot": self._format_snapshot(snapshot),
            },
            status=status.HTTP_200_OK,
        )

    @action(
        detail=True,
        methods=["delete"],
        url_path="snapshots/(?P<snapshot_id>[^/.]+)",
    )
    def delete_snapshot(self, request, pk=None, snapshot_id=None):
        project = self.get_object()
        snapshot = get_object_or_404(ProjectSnapshot, id=snapshot_id, project=project)
        self._delete_snapshot(snapshot)
        return Response(status=status.HTTP_204_NO_CONTENT)
