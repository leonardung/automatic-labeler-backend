"""
OCR Image View with GPU Service Integration

This is a modified version of ocr_image_view.py that delegates GPU operations
to the GPU microservice instead of running PaddleOCR locally.

INSTRUCTIONS:
1. Review this file
2. If satisfied, rename the original ocr_image_view.py to ocr_image_view_local.py (backup)
3. Rename this file to ocr_image_view.py
4. Update imports in urls.py if needed
"""

import logging
import uuid
from pathlib import Path

from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.conf import settings

from api.models.image import ImageModel
from api.models.mask import MaskCategory
from api.models.ocr import OcrAnnotation
from api.models.project import Project
from api.serializers import ImageModelSerializer
from api.services.gpu_service_client import get_gpu_client, GPUServiceError
from .base_image_view import BaseImageViewSet

log = logging.getLogger(__name__)

DEFAULT_KIE_CATEGORIES = ["header", "field", "value", "table"]
DEFAULT_KIE_MAX_SEQ_LEN = 512
DEFAULT_KIE_MIN_OVERLAP = 64


class OcrImageViewSet(BaseImageViewSet):
    """
    OCR/KIE specific endpoints using GPU microservice.
    """

    allowed_project_types = ("ocr", "ocr_kie")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_client = get_gpu_client()

    @action(detail=True, methods=["post"])
    def detect_regions(self, request, pk=None):
        """
        Run PaddleOCR text detection on the image via GPU service.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error

        model_name = (
            request.data.get("model_name") or request.query_params.get("model_name")
        )

        # Get configuration from project
        config = image.project.ocr_model_config
        if isinstance(config, dict):
            det_config = config.get("det")
            tolerance_ratio = self._parse_float(config.get("tolerance_ratio"), 0.2)
        else:
            det_config = None
            tolerance_ratio = 0.2

        # Call GPU service
        try:
            shapes = self.gpu_client.detect_regions(
                image_path=image.image.path,
                model_name=model_name,
                config=det_config,
            )
        except GPUServiceError as exc:
            log.exception("GPU service detection failed: %s", exc)
            return Response(
                {"error": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Store annotations
        stored = self._replace_annotations(image, shapes)
        return Response({"shapes": stored}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"])
    def recognize_text(self, request, pk=None):
        """
        Run PaddleOCR text recognition via GPU service.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error

        model_name = (
            request.data.get("model_name") or request.query_params.get("model_name")
        )
        shapes = request.data.get("shapes") or []
        if not shapes:
            return Response({"shapes": []}, status=status.HTTP_200_OK)

        # Call GPU service
        try:
            result = self.gpu_client.recognize_text(
                image_path=image.image.path,
                shapes=shapes,
                model_name=model_name,
            )
            recognized_shapes = result.get("shapes", [])
        except GPUServiceError as exc:
            log.exception("GPU service recognition failed: %s", exc)
            return Response(
                {"error": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Get categories for KIE projects
        categories = []
        if image.project.type == "ocr_kie":
            categories = self._resolve_kie_categories(request, image)

        # Store annotations
        saved = self._upsert_annotations(image, recognized_shapes)
        response_payload = {"shapes": saved}
        if categories:
            response_payload["categories"] = categories

        return Response(response_payload, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"])
    def classify_kie(self, request, pk=None):
        """
        Run SER/KIE classification via GPU service.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error

        shapes = request.data.get("shapes") or []
        if not shapes:
            return Response(
                {"error": "Detection results are required before classification."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Parse parameters
        checkpoint_type_raw = (
            request.data.get("checkpoint_type")
            or request.query_params.get("checkpoint_type")
            or "best"
        )
        checkpoint_type = str(checkpoint_type_raw).lower()
        if checkpoint_type not in ("best", "latest"):
            checkpoint_type = "best"

        model_name = (
            request.data.get("model_name") or request.query_params.get("model_name")
        )

        use_gpu_raw = request.data.get("use_gpu")
        if use_gpu_raw is None:
            use_gpu_raw = request.query_params.get("use_gpu")
        use_gpu = (
            True
            if use_gpu_raw is None
            else str(use_gpu_raw).lower() in ("1", "true", "yes", "on")
        )

        max_len_raw = request.data.get("max_len_per_part") or request.query_params.get(
            "max_len_per_part"
        )
        min_overlap_raw = (
            request.data.get("min_overlap")
            or request.data.get("min_token_overlap")
            or request.query_params.get("min_overlap")
            or request.query_params.get("min_token_overlap")
        )

        try:
            max_len_per_part = (
                int(max_len_raw) if max_len_raw is not None else DEFAULT_KIE_MAX_SEQ_LEN
            )
            min_overlap = (
                int(min_overlap_raw)
                if min_overlap_raw is not None
                else DEFAULT_KIE_MIN_OVERLAP
            )
        except (TypeError, ValueError):
            return Response(
                {"error": "Invalid KIE chunking parameters."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if max_len_per_part <= 2 or min_overlap < 0 or max_len_per_part <= min_overlap:
            return Response(
                {"error": "Invalid KIE chunking parameters."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Call GPU service
        try:
            result = self.gpu_client.classify_kie(
                image_path=image.image.path,
                shapes=shapes,
                project_id=image.project.id,
                model_name=model_name,
                checkpoint_type=checkpoint_type,
                use_gpu=use_gpu,
                max_len_per_part=max_len_per_part,
                min_overlap=min_overlap,
            )
            classified_shapes = result.get("shapes", [])
            categories = result.get("categories", [])
        except GPUServiceError as exc:
            log.exception("GPU service KIE classification failed: %s", exc)
            error_str = str(exc)
            if "not found" in error_str.lower():
                status_code = status.HTTP_400_BAD_REQUEST
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return Response({"error": error_str}, status=status_code)

        # Store annotations
        saved = self._upsert_annotations(image, classified_shapes)
        return Response(
            {"shapes": saved, "categories": categories},
            status=status.HTTP_200_OK,
        )

    @action(detail=True, methods=["post"])
    def ocr_annotations(self, request, pk=None):
        """
        Create or update OCR annotations in bulk.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error
        shapes = request.data.get("shapes") or []
        saved = self._upsert_annotations(image, shapes)
        return Response({"shapes": saved}, status=status.HTTP_200_OK)

    @ocr_annotations.mapping.delete
    def delete_ocr_annotations(self, request, pk=None):
        """
        Delete one or more OCR annotations by id.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error
        ids = request.data.get("ids") or []
        qs = OcrAnnotation.objects.filter(image=image)
        if ids:
            qs = qs.filter(id__in=ids)
        deleted, _ = qs.delete()
        return Response({"deleted": deleted}, status=status.HTTP_200_OK)

    @action(detail=False, methods=["post"])
    def configure_models(self, request):
        """
        Configure models via GPU service.
        """
        project_id = request.data.get("project_id")
        model_config = request.data.get("model_config")
        detect_model = request.data.get("detect_model")
        recognize_model = request.data.get("recognize_model")
        classify_model = request.data.get("classify_model")

        try:
            result = self.gpu_client.configure_models(
                detect_model=detect_model,
                recognize_model=recognize_model,
                classify_model=classify_model,
                model_config=model_config,
            )

            # Persist config to database if project_id provided
            if project_id and isinstance(model_config, dict):
                try:
                    project = Project.objects.get(id=project_id, user=request.user)
                    project.ocr_model_config = model_config
                    project.save(update_fields=["ocr_model_config"])
                except Project.DoesNotExist:
                    pass
                except Exception as exc:
                    log.exception("Failed to persist OCR model config: %s", exc)

            return Response(result, status=status.HTTP_200_OK)
        except GPUServiceError as exc:
            log.exception("GPU service model configuration failed: %s", exc)
            return Response(
                {"error": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def bulk_update_validation(self, request):
        """
        Bulk update is_label field for multiple images.
        """
        image_ids = request.data.get("image_ids", [])
        is_label = request.data.get("is_label")

        if not isinstance(image_ids, list) or not image_ids:
            return Response(
                {"error": "image_ids must be a non-empty list."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not isinstance(is_label, bool):
            return Response(
                {"error": "is_label must be a boolean."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        queryset = self.get_queryset().filter(id__in=image_ids)
        updated_count = queryset.update(is_label=is_label)
        updated_images = queryset.all()
        serializer = self.get_serializer(updated_images, many=True)

        return Response(
            {
                "updated_count": updated_count,
                "images": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    @action(detail=False, methods=["post"])
    def bulk_clear_annotations(self, request):
        """
        Bulk clear OCR annotations for multiple images.
        """
        image_ids = request.data.get("image_ids", [])

        if not isinstance(image_ids, list) or not image_ids:
            return Response(
                {"error": "image_ids must be a non-empty list."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        queryset = self.get_queryset().filter(id__in=image_ids)
        deleted_count = 0
        for image in queryset:
            count, _ = OcrAnnotation.objects.filter(image=image).delete()
            deleted_count += count

        return Response(
            {
                "deleted_count": deleted_count,
                "cleared_image_count": queryset.count(),
            },
            status=status.HTTP_200_OK,
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _validate_project(self, image: ImageModel):
        """Validate project type"""
        if (
            self.allowed_project_types
            and image.project.type not in self.allowed_project_types
        ):
            return Response(
                {
                    "error": f"Endpoint only supports {self.allowed_project_types} projects."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        return None

    def _resolve_kie_categories(self, request, image: ImageModel) -> list[str]:
        """Resolve KIE categories from request or project"""
        incoming_categories = request.data.get("categories") or []
        cleaned_categories = [
            c for c in incoming_categories if isinstance(c, str) and c
        ]
        if cleaned_categories:
            return cleaned_categories

        project_categories = list(
            MaskCategory.objects.filter(project=image.project).values_list(
                "name", flat=True
            )
        )
        if project_categories:
            return list(project_categories)
        return DEFAULT_KIE_CATEGORIES.copy()

    @staticmethod
    def _shape_from_annotation(annotation: OcrAnnotation):
        """Convert annotation to shape dict"""
        return {
            "id": str(annotation.id),
            "type": annotation.shape_type,
            "points": annotation.points,
            "text": annotation.text or "",
            "category": annotation.category,
        }

    def _replace_annotations(self, image: ImageModel, shapes_payload):
        """Replace all annotations for an image"""
        OcrAnnotation.objects.filter(image=image).delete()
        created = []
        for shape in shapes_payload:
            obj = OcrAnnotation.objects.create(
                image=image,
                shape_type=shape.get("type", "rect"),
                points=shape.get("points", []),
                text=shape.get("text", "") or "",
                category=shape.get("category"),
            )
            created.append(obj)
        return [self._shape_from_annotation(obj) for obj in created]

    def _upsert_annotations(self, image: ImageModel, shapes_payload):
        """Upsert annotations for an image"""
        serialized = []
        for shape in shapes_payload:
            shape_id = shape.get("id")
            instance = None
            if shape_id:
                try:
                    instance = OcrAnnotation.objects.get(id=shape_id, image=image)
                except (OcrAnnotation.DoesNotExist, ValueError):
                    instance = None
            if instance is None:
                instance = OcrAnnotation(image=image)
            instance.shape_type = shape.get("type", instance.shape_type or "rect")
            instance.points = shape.get("points", instance.points or [])
            instance.text = shape.get("text", instance.text or "") or ""
            instance.category = shape.get("category", instance.category)
            instance.save()
            serialized.append(self._shape_from_annotation(instance))
        return serialized

    @staticmethod
    def _parse_float(value, default: float) -> float:
        """Parse float with default"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


# NOTE: The following endpoints from the original file are not yet migrated:
# - upload_dataset
# - dataset_progress
# - model_config
# - configure_trained_models
#
# These can remain in the Django backend as they primarily deal with database
# operations and file management, not GPU-intensive OCR operations.
