import uuid

from PIL import Image
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response

from api.models.image import ImageModel
from api.models.ocr import OcrAnnotation
from api.serializers import ImageModelSerializer
from .base_image_view import BaseImageViewSet


class OcrImageViewSet(BaseImageViewSet):
    """
    OCR/KIE specific endpoints: region detection, text recognition, and annotation CRUD.
    """

    allowed_project_types = ("ocr", "ocr_kie")

    @action(detail=True, methods=["post"])
    def detect_regions(self, request, pk=None):
        """
        Mock OCR region detector that returns a few sample boxes/polygons.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error
        width, height = self._get_image_dimensions(image.image.path)
        shapes = [
            self._mock_rect_shape(width, height, 0.08, 0.1, 0.48, 0.24),
            self._mock_rect_shape(width, height, 0.52, 0.15, 0.9, 0.3),
            self._mock_polygon_shape(
                [
                    (0.12 * width, 0.42 * height),
                    (0.46 * width, 0.4 * height),
                    (0.52 * width, 0.47 * height),
                    (0.18 * width, 0.5 * height),
                ],
                text="",
            ),
        ]
        stored = self._replace_annotations(image, shapes)
        return Response({"shapes": stored}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"])
    def recognize_text(self, request, pk=None):
        """
        Mock OCR recognizer that fills in placeholder text per shape.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error
        shapes = request.data.get("shapes") or []
        if not shapes:
            return Response({"shapes": []}, status=status.HTTP_200_OK)
        recognized_payload = []
        for idx, shape in enumerate(shapes):
            text = shape.get("text") or f"Text #{idx + 1}"
            recognized_payload.append({**shape, "text": text})

        saved = self._upsert_annotations(image, recognized_payload)
        return Response({"shapes": saved}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"])
    def classify_kie(self, request, pk=None):
        """
        Mock KIE classifier that assigns categories in a round-robin fashion.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error
        shapes = request.data.get("shapes") or []
        incoming_categories = request.data.get("categories") or []
        fallback_categories = incoming_categories or ["header", "field", "value", "table"]

        classified = []
        for idx, shape in enumerate(shapes):
            category = shape.get("category") or fallback_categories[idx % len(fallback_categories)]
            classified.append({**shape, "category": category})

        saved = self._upsert_annotations(image, classified)
        return Response(
            {"shapes": saved, "categories": fallback_categories},
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
        Delete one or more OCR annotations by id. If no ids supplied, delete all.
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

    def _validate_project(self, image: ImageModel):
        if self.allowed_project_types and image.project.type not in self.allowed_project_types:
            return Response(
                {"error": f"Endpoint only supports {self.allowed_project_types} projects."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return None

    @staticmethod
    def _mock_rect_shape(width: int, height: int, x1: float, y1: float, x2: float, y2: float, text=""):
        return {
            "id": uuid.uuid4().hex,
            "type": "rect",
            "points": [
                {"x": int(x1 * width), "y": int(y1 * height)},
                {"x": int(x2 * width), "y": int(y1 * height)},
                {"x": int(x2 * width), "y": int(y2 * height)},
                {"x": int(x1 * width), "y": int(y2 * height)},
            ],
            "text": text,
            "category": None,
        }

    @staticmethod
    def _mock_polygon_shape(points, text=""):
        return {
            "id": uuid.uuid4().hex,
            "type": "polygon",
            "points": [{"x": int(x), "y": int(y)} for x, y in points],
            "text": text,
            "category": None,
        }

    @staticmethod
    def _shape_from_annotation(annotation: OcrAnnotation):
        return {
            "id": str(annotation.id),
            "type": annotation.shape_type,
            "points": annotation.points,
            "text": annotation.text or "",
            "category": annotation.category,
        }

    def _replace_annotations(self, image: ImageModel, shapes_payload):
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
    def _get_image_dimensions(image_path: str):
        with Image.open(image_path) as img:
            return img.size
