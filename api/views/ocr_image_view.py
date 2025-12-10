import inspect
import logging
import sys
import uuid
from pathlib import Path

from PIL import Image
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response

from api.models.image import ImageModel
from api.models.ocr import OcrAnnotation
from api.serializers import ImageModelSerializer
from .base_image_view import BaseImageViewSet


log = logging.getLogger(__name__)
PADDLE_SUBMODULE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "submodules" / "PaddleOCR"
)
if str(PADDLE_SUBMODULE_PATH) not in sys.path:
    sys.path.append(str(PADDLE_SUBMODULE_PATH))
try:
    from paddleocr import TextDetection
except Exception as exc:  # pragma: no cover - import guard for optional dependency
    TextDetection = None
    log.exception("Failed to import paddleocr TextDetection: %s", exc)

DEFAULT_DET_MODEL_NAME = "PP-OCRv5_server_det"
_DETECTOR_CACHE: dict[str, TextDetection] = {}


class OcrImageViewSet(BaseImageViewSet):
    """
    OCR/KIE specific endpoints: region detection, text recognition, and annotation CRUD.
    """

    allowed_project_types = ("ocr", "ocr_kie")

    @action(detail=True, methods=["post"])
    def detect_regions(self, request, pk=None):
        """
        Run PaddleOCR text detection on the image and store polygons as annotations.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error

        if TextDetection is None:
            return Response(
                {
                    "error": "PaddleOCR TextDetection unavailable. Is the submodule installed?"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        model_name = (
            request.data.get("model_name")
            or request.query_params.get("model_name")
            or DEFAULT_DET_MODEL_NAME
        )
        score_threshold_raw = request.data.get(
            "score_threshold"
        ) or request.query_params.get("score_threshold")
        try:
            score_threshold = (
                float(score_threshold_raw) if score_threshold_raw is not None else 0.3
            )
        except (TypeError, ValueError):
            score_threshold = 0.3

        detector = _DETECTOR_CACHE.get(model_name)
        if detector is None:
            try:
                detector = TextDetection(model_name=model_name)
                _DETECTOR_CACHE[model_name] = detector
            except Exception as exc:  # pragma: no cover - runtime dependency
                log.exception("Failed to initialize PaddleOCR detector: %s", exc)
                return Response(
                    {"error": f"Failed to load detector model '{model_name}'."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        try:
            output = detector.predict(image.image.path, batch_size=1)
        except Exception as exc:  # pragma: no cover - runtime dependency
            log.exception("PaddleOCR detection failed for image %s: %s", image.id, exc)
            return Response(
                {"error": "Region detection failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        shapes = []
        if output:
            result = output[0]
            polys = result.get("dt_polys", [])
            scores = result.get("dt_scores", [])
            for idx, poly in enumerate(polys):
                score = scores[idx] if idx < len(scores) else None
                if score is not None and score < score_threshold:
                    continue
                points = [{"x": int(pt[0]), "y": int(pt[1])} for pt in poly]
                rect_points = self._polygon_to_rect(points, tolerance_ratio=10)
                shapes.append(
                    {
                        "id": uuid.uuid4().hex,
                        "type": "rect" if rect_points else "polygon",
                        "points": rect_points or points,
                        "text": "",
                        "category": None,
                    }
                )

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
        fallback_categories = incoming_categories or [
            "header",
            "field",
            "value",
            "table",
        ]

        classified = []
        for idx, shape in enumerate(shapes):
            category = (
                shape.get("category")
                or fallback_categories[idx % len(fallback_categories)]
            )
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

    @staticmethod
    def _mock_rect_shape(
        width: int, height: int, x1: float, y1: float, x2: float, y2: float, text=""
    ):
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

    @staticmethod
    def _polygon_to_rect(points, tolerance_ratio: float = 0.1):
        """
        If points roughly form an axis-aligned rectangle, return rectangle corners.
        """
        if len(points) != 4:
            return None
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return None

        bbox_area = width * height
        poly_area = OcrImageViewSet._polygon_area(points)
        if poly_area <= 0:
            return None

        # Check area closeness to rectangle area
        if poly_area / bbox_area < 0.5:
            return None

        tol = max(width, height) * tolerance_ratio
        for x, y in zip(xs, ys):
            if (abs(x - min_x) > tol and abs(x - max_x) > tol) or (
                abs(y - min_y) > tol and abs(y - max_y) > tol
            ):
                return None

        return [
            {"x": min_x, "y": min_y},
            {"x": max_x, "y": min_y},
            {"x": max_x, "y": max_y},
            {"x": min_x, "y": max_y},
        ]

    @staticmethod
    def _polygon_area(points):
        area = 0.0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]["x"], points[i]["y"]
            x2, y2 = points[(i + 1) % n]["x"], points[(i + 1) % n]["y"]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
