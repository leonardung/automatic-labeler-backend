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
_paddle_ocr_detector = None
_paddle_ocr_use_predict = False
PADDLE_SUBMODULE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "submodules" / "PaddleOCR"
)
if str(PADDLE_SUBMODULE_PATH) not in sys.path:
    sys.path.append(str(PADDLE_SUBMODULE_PATH))


class OcrImageViewSet(BaseImageViewSet):
    """
    OCR/KIE specific endpoints: region detection, text recognition, and annotation CRUD.
    """

    allowed_project_types = ("ocr", "ocr_kie")

    @action(detail=True, methods=["post"])
    def detect_regions(self, request, pk=None):
        """
        Run PaddleOCR text detection (PP-OCRv5_mobile_det) and store polygons.
        """
        image: ImageModel = self.get_object()
        error = self._validate_project(image)
        if error:
            return error
        try:
            detector, use_predict = self._get_paddle_detector()
        except Exception as exc:
            log.exception("Failed to initialize PaddleOCR detector")
            return Response(
                {"error": f"Failed to initialize PaddleOCR detector: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        try:
            if use_predict:
                raw_result = detector.predict(image.image.path)
            else:  # Backwards compatibility with older PaddleOCR interface
                raw_result = detector.ocr(
                    image.image.path, det=True, rec=False, cls=False
                )
            polygons = self._parse_detection_polygons(raw_result)
        except Exception as exc:  # pragma: no cover - relies on external model runtime
            log.exception("PaddleOCR detection failed for image %s", image.id)
            return Response(
                {"error": f"PaddleOCR detection failed: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        if not polygons:
            OcrAnnotation.objects.filter(image=image).delete()
            return Response({"shapes": []}, status=status.HTTP_200_OK)

        shapes = [
            {
                "id": uuid.uuid4().hex,
                "type": "polygon",
                "points": poly,
                "text": "",
                "category": None,
            }
            for poly in polygons
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

    def _get_paddle_detector(self):
        """
        Lazily instantiate a PaddleOCR detector using the PP-OCRv5_mobile_det model.
        Supports both the new PaddleOCR (PaddleX-backed) API and the legacy API.
        """
        global _paddle_ocr_detector, _paddle_ocr_use_predict
        if _paddle_ocr_detector is not None:
            return _paddle_ocr_detector, _paddle_ocr_use_predict

        try:

            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install paddlepaddle and paddleocr to enable detection."
            ) from exc

        init_params = inspect.signature(PaddleOCR.__init__).parameters
        detector_kwargs = {}
        # New (PaddleX) API exposes text_detection_model_name
        if "text_detection_model_name" in init_params:
            detector_kwargs.update(
                {
                    "text_detection_model_name": "PP-OCRv5_mobile_det",
                    "text_recognition_model_name": None,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                }
            )
            if "ocr_version" in init_params:
                detector_kwargs["ocr_version"] = "PP-OCRv5"
            if "lang" in init_params:
                detector_kwargs["lang"] = "en"
            _paddle_ocr_use_predict = True
        else:
            # Legacy API: disable recognition/cls to keep only detection.
            if "ocr_version" in init_params:
                detector_kwargs["ocr_version"] = "PP-OCRv5"
            if "lang" in init_params:
                detector_kwargs["lang"] = "en"
            if "use_angle_cls" in init_params:
                detector_kwargs["use_angle_cls"] = False
            detector_kwargs.update({"rec": False, "det": True, "cls": False})
            _paddle_ocr_use_predict = False

        _paddle_ocr_detector = PaddleOCR(**detector_kwargs)
        return _paddle_ocr_detector, _paddle_ocr_use_predict

    @staticmethod
    def _parse_detection_polygons(det_result):
        """
        Normalize PaddleOCR detection output (old or new API) into polygon point lists.
        """

        def to_python(value):
            try:
                import numpy as np  # type: ignore
            except (
                Exception
            ):  # pragma: no cover - numpy should be present via requirements
                np = None  # type: ignore
            if np is not None and isinstance(value, np.ndarray):
                return value.tolist()
            return value

        def looks_like_polygon(val):
            val = to_python(val)
            if isinstance(val, (list, tuple)):
                if len(val) == 8 and all(isinstance(v, (int, float)) for v in val):
                    return True
                if len(val) >= 4 and all(
                    isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in val
                ):
                    return True
            return False

        def normalize_polygon(val):
            val = to_python(val)
            points = []
            if (
                isinstance(val, (list, tuple))
                and len(val) == 8
                and all(isinstance(v, (int, float)) for v in val)
            ):
                coords = list(val)
                points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            elif isinstance(val, (list, tuple)):
                for pt in val:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        points.append((pt[0], pt[1]))
            normalized = [
                {"x": int(round(x)), "y": int(round(y))}
                for x, y in points
                if x is not None and y is not None
            ]
            return normalized if len(normalized) >= 4 else None

        polygons = []
        queue = [det_result]
        while queue:
            item = to_python(queue.pop(0))
            if looks_like_polygon(item):
                normalized = normalize_polygon(item)
                if normalized:
                    polygons.append(normalized)
                continue
            if isinstance(item, dict):
                for key in (
                    "boxes",
                    "polygons",
                    "points",
                    "det_polygons",
                    "det_boxes",
                    "bbox",
                    "bbox_points",
                    "result",
                ):
                    if key in item and item[key] is not None:
                        queue.append(item[key])
                continue
            if isinstance(item, (list, tuple)):
                if len(item) == 2 and looks_like_polygon(item[0]):
                    normalized = normalize_polygon(item[0])
                    if normalized:
                        polygons.append(normalized)
                    continue
                queue.extend(item)
                continue
            if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
                queue.extend(list(item))

        return polygons

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
