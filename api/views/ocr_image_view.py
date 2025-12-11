import inspect
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
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
    from paddleocr import TextDetection, TextRecognition
except Exception as exc:  # pragma: no cover - import guard for optional dependency
    TextDetection = None
    TextRecognition = None
    log.exception("Failed to import paddleocr: %s", exc)

DEFAULT_DET_MODEL_NAME = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL_NAME = "PP-OCRv5_server_rec"
_DETECTOR_CACHE: dict[str, TextDetection] = {}
_RECOGNIZER_CACHE: dict[str, TextRecognition] = {}
ACTIVE_DET_MODEL_NAME = DEFAULT_DET_MODEL_NAME
ACTIVE_REC_MODEL_NAME = DEFAULT_REC_MODEL_NAME


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
            or ACTIVE_DET_MODEL_NAME
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
        tolerance_raw = request.data.get("tolerance_ratio") or request.query_params.get(
            "tolerance_ratio"
        )
        try:
            tolerance_ratio = float(tolerance_raw) if tolerance_raw is not None else 0
        except (TypeError, ValueError):
            tolerance_ratio = 0.2

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
                rect_points = self._polygon_to_rect(
                    points, tolerance_ratio=tolerance_ratio
                )
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
        Run PaddleOCR text recognition on provided shapes (cropped regions).
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error

        if TextRecognition is None:
            return Response(
                {
                    "error": "PaddleOCR TextRecognition unavailable. Is the submodule installed?"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        model_name = (
            request.data.get("model_name")
            or request.query_params.get("model_name")
            or ACTIVE_REC_MODEL_NAME
        )
        recognizer = _RECOGNIZER_CACHE.get(model_name)
        if recognizer is None:
            try:
                recognizer = TextRecognition(model_name=model_name)
                _RECOGNIZER_CACHE[model_name] = recognizer
            except Exception as exc:  # pragma: no cover - runtime dependency
                log.exception("Failed to initialize PaddleOCR recognizer: %s", exc)
                return Response(
                    {"error": f"Failed to load recognition model '{model_name}'."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        shapes = request.data.get("shapes") or []
        if not shapes:
            return Response({"shapes": []}, status=status.HTTP_200_OK)

        try:
            with Image.open(image.image.path) as img:
                base_image = img.convert("RGB")
                recognized_payload = []
                img_width, img_height = base_image.size
                for shape in shapes:
                    points = shape.get("points") or []
                    bbox = self._bbox_from_points(points)
                    if bbox is None:
                        recognized_payload.append(
                            {**shape, "text": shape.get("text") or ""}
                        )
                        continue
                    min_x, min_y, max_x, max_y = bbox
                    min_x = max(0, min_x)
                    min_y = max(0, min_y)
                    max_x = min(img_width, max_x)
                    max_y = min(img_height, max_y)
                    if min_x >= max_x or min_y >= max_y:
                        recognized_payload.append(
                            {**shape, "text": shape.get("text") or ""}
                        )
                        continue
                    region = base_image.crop((min_x, min_y, max_x, max_y))
                    crop_arr = np.array(region)
                    text = shape.get("text") or ""
                    try:
                        rec_output = recognizer.predict(input=crop_arr, batch_size=1)
                        if rec_output:
                            rec_text = rec_output[0].get("rec_text", "")
                            text = rec_text or text
                    except Exception as exc:  # pragma: no cover - runtime dependency
                        log.exception(
                            "Text recognition failed for shape on image %s: %s",
                            image.id,
                            exc,
                        )
                    recognized_payload.append({**shape, "text": text})
        except Exception as exc:  # pragma: no cover - runtime dependency
            log.exception(
                "Failed to process image for recognition %s: %s", image.id, exc
            )
            return Response(
                {"error": "Text recognition failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

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

        poly_area = OcrImageViewSet._polygon_area(points)
        if poly_area <= 0:
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

    @staticmethod
    def _bbox_from_points(points):
        if not points:
            return None
        try:
            xs = [int(p["x"]) for p in points]
            ys = [int(p["y"]) for p in points]
        except (TypeError, KeyError, ValueError):
            return None
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if min_x == max_x or min_y == max_y:
            return None
        return (min_x, min_y, max_x, max_y)

    @action(detail=False, methods=["post"])
    def configure_models(self, request):
        """
        Load and set active detect/recognize models. Unload previous active models if changed.
        """
        global ACTIVE_DET_MODEL_NAME, ACTIVE_REC_MODEL_NAME

        if TextDetection is None or TextRecognition is None:
            return Response(
                {"error": "PaddleOCR modules unavailable. Is the submodule installed?"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        detect_model = request.data.get("detect_model") or ACTIVE_DET_MODEL_NAME
        recognize_model = request.data.get("recognize_model") or ACTIVE_REC_MODEL_NAME

        changed = {"detect": False, "recognize": False}

        if detect_model != ACTIVE_DET_MODEL_NAME or detect_model not in _DETECTOR_CACHE:
            _DETECTOR_CACHE.pop(ACTIVE_DET_MODEL_NAME, None)
            try:
                _DETECTOR_CACHE[detect_model] = TextDetection(model_name=detect_model)
                ACTIVE_DET_MODEL_NAME = detect_model
                changed["detect"] = True
            except Exception as exc:  # pragma: no cover - runtime dependency
                log.exception("Failed to load detect model %s: %s", detect_model, exc)
                return Response(
                    {"error": f"Failed to load detect model '{detect_model}'."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        if (
            recognize_model != ACTIVE_REC_MODEL_NAME
            or recognize_model not in _RECOGNIZER_CACHE
        ):
            _RECOGNIZER_CACHE.pop(ACTIVE_REC_MODEL_NAME, None)
            try:
                _RECOGNIZER_CACHE[recognize_model] = TextRecognition(
                    model_name=recognize_model
                )
                ACTIVE_REC_MODEL_NAME = recognize_model
                changed["recognize"] = True
            except Exception as exc:  # pragma: no cover - runtime dependency
                log.exception(
                    "Failed to load recognize model %s: %s", recognize_model, exc
                )
                return Response(
                    {"error": f"Failed to load recognize model '{recognize_model}'."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(
            {
                "detect_model": ACTIVE_DET_MODEL_NAME,
                "recognize_model": ACTIVE_REC_MODEL_NAME,
                "changed": changed,
            },
            status=status.HTTP_200_OK,
        )
