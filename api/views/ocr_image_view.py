import math
import json
import logging
import subprocess
import sys
import uuid
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.conf import settings

from api.models.image import ImageModel
from api.models.mask import MaskCategory
from api.models.ocr import OcrAnnotation
from api.models.project import Project
from api.models.training import TrainingRun
from api.serializers import ImageModelSerializer
from api.views.ocr import helper as ocr_helper
from .base_image_view import BaseImageViewSet


log = logging.getLogger(__name__)
PADDLE_ROOT = Path(settings.BASE_DIR) / "submodules" / "PaddleOCR"
SUBMODULE_ROOT = Path(settings.BASE_DIR) / "submodules"
sys.path.append(str(PADDLE_ROOT))
sys.path.append(str(SUBMODULE_ROOT))

from paddleocr import TextDetection, TextRecognition
from PaddleOCR.tools import infer_kie_token_ser, program
from paddlenlp.transformers import LayoutLMv2Tokenizer

DEFAULT_DET_MODEL_NAME = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL_NAME = "PP-OCRv5_server_rec"
_DETECTOR_CACHE: dict[str, TextDetection] = {}
_RECOGNIZER_CACHE: dict[str, TextRecognition] = {}
ACTIVE_DET_MODEL_NAME = DEFAULT_DET_MODEL_NAME
ACTIVE_REC_MODEL_NAME = DEFAULT_REC_MODEL_NAME
ACTIVE_KIE_MODEL_NAME: str | None = None
DEFAULT_KIE_CATEGORIES = ["header", "field", "value", "table"]
DEFAULT_KIE_MAX_SEQ_LEN = 512
DEFAULT_KIE_MIN_OVERLAP = 64
_TRAINED_INFERENCE_SUBDIR = "inference"
_TRAINED_MODEL_KEY = "trained-project-{project_id}-{target}"
_DATASET_PROGRESS: dict[tuple[int, int], dict] = {}


def _trained_model_dir(project_id: int | str, target: str) -> Path:
    return ocr_helper.MEDIA_PROJECT_ROOT / str(project_id) / "models" / target


def _latest_model_dir(project_id: int | str, target: str) -> Path | None:
    """
    Return the most recent models/<job>/<target> directory for a project, if any.
    """
    base = ocr_helper.MEDIA_PROJECT_ROOT / str(project_id) / "models"
    if not base.exists():
        return None
    candidates: list[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        cand = child / target
        if cand.exists() and cand.is_dir():
            candidates.append(cand)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_models_dir(
    path_str: str | None, project_id: int | str, target: str
) -> Path:
    """
    Normalize a stored models_dir to an existing path, handling Windows->WSL paths.
    """
    base_dir = _trained_model_dir(project_id, target)
    latest_dir = _latest_model_dir(project_id, target)
    if not path_str:
        return latest_dir or base_dir

    candidates: list[Path] = []
    raw_path = Path(path_str)
    candidates.append(raw_path)

    # If the stored path was relative, try resolving it from the expected base.
    if not raw_path.is_absolute():
        candidates.append((base_dir / raw_path).resolve())

    # Translate Windows drive paths (e.g., C:\\foo\\bar) into WSL-style /mnt/c/foo/bar.
    if ":" in path_str and "\\" in path_str:
        drive, _, remainder = path_str.partition(":")
        translated = (
            Path("/mnt") / drive.lower() / remainder.lstrip("\\/").replace("\\", "/")
        )
        candidates.append(translated)

    # Normalize backslashes to forward slashes as another fallback.
    if "\\" in path_str:
        candidates.append(Path(path_str.replace("\\", "/")))

    # Try the most recent job subdir as a final fallback.
    if latest_dir:
        candidates.append(latest_dir)
    # Always include the canonical location as a last resort.
    candidates.append(base_dir)

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _find_checkpoint_prefix(models_dir: Path, prefer_best: bool = True) -> Path:
    """
    Return the checkpoint prefix (without extension) for available weights.
    Preference order (best): best_accuracy, best_model/model, latest, any *.pdparams.
    Preference order (latest): latest, latest/model, latest/model_state, best_accuracy, best_model/model, any *.pdparams.
    """
    if prefer_best:
        candidates = [
            models_dir / "best_accuracy",
            models_dir / "best_model" / "model",
            models_dir / "latest",
            models_dir / "latest" / "model",
            models_dir / "latest" / "model_state",
        ]
    else:
        candidates = [
            models_dir / "latest",
            models_dir / "latest" / "model",
            models_dir / "latest" / "model_state",
            models_dir / "best_accuracy",
            models_dir / "best_model" / "model",
        ]
    for prefix in candidates:
        if prefix.with_suffix(".pdparams").exists():
            return prefix

    pdparams_files = sorted(models_dir.rglob("*.pdparams"))
    if not pdparams_files:
        raise FileNotFoundError(f"No checkpoint files found in {models_dir}")
    return pdparams_files[0].with_suffix("")


def _has_inference_files(inference_dir: Path) -> bool:
    if not inference_dir.is_dir():
        return False
    has_model_file = any(inference_dir.glob("*.pdmodel")) or any(
        inference_dir.glob("*.json")
    )
    has_params = any(inference_dir.glob("*.pdiparams"))
    return has_model_file and has_params


def _export_trained_model(
    project_id: int | str,
    target: str,
    run_id: str | None = None,
    checkpoint_type: str = "best",
) -> tuple[Path, Path, TrainingRun | None]:
    """
    Export a trained PaddleOCR model to inference format.

    Returns (inference_dir, checkpoint_prefix, training_run_used).
    """
    prefer_best = checkpoint_type != "latest"
    run: TrainingRun | None = None
    run_qs = TrainingRun.objects.filter(project_id=project_id, target=target)
    if run_id:
        run_qs = run_qs.filter(id=run_id)
    # Prefer completed runs first, then fallback to the latest any-status run.
    run = (
        run_qs.filter(status="completed")
        .order_by("-finished_at", "-created_at")
        .first()
        or run_qs.order_by("-finished_at", "-created_at").first()
    )

    models_dir = _resolve_models_dir(
        run.models_dir if run else None, project_id=project_id, target=target
    )
    if not models_dir.exists():
        raise FileNotFoundError(
            f"No trained {target} model directory found for project {project_id}."
        )

    config_path = models_dir.parent / "kie/config.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found for trained {target} model at {config_path}."
        )

    checkpoint_prefix: Path
    run_checkpoint = None
    if run:
        run_checkpoint = run.best_checkpoint if prefer_best else run.latest_checkpoint
    if run_checkpoint:
        candidate = Path(run_checkpoint)
        if candidate.with_suffix(".pdparams").exists():
            checkpoint_prefix = candidate
        else:
            checkpoint_prefix = _find_checkpoint_prefix(
                models_dir, prefer_best=prefer_best
            )
    else:
        checkpoint_prefix = _find_checkpoint_prefix(models_dir, prefer_best=prefer_best)
    inference_dir = models_dir / _TRAINED_INFERENCE_SUBDIR
    checkpoint_path = checkpoint_prefix.with_suffix(".pdparams")

    needs_export = True
    if _has_inference_files(inference_dir) and checkpoint_path.exists():
        try:
            checkpoint_mtime = checkpoint_path.stat().st_mtime
            infer_mtime = max(p.stat().st_mtime for p in inference_dir.glob("*.pd*"))
            needs_export = checkpoint_mtime > infer_mtime
        except ValueError:
            needs_export = True

    if not needs_export:
        return inference_dir, checkpoint_prefix, run

    cmd = [
        sys.executable,
        "tools/export_model.py",
        "-c",
        str(config_path),
        "-o",
        f"Global.checkpoints={checkpoint_prefix.as_posix()}",
        f"Global.save_inference_dir={inference_dir.as_posix()}",
    ]
    process = subprocess.run(
        cmd,
        cwd=ocr_helper.PADDLE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        err_output = process.stderr or process.stdout
        raise RuntimeError(
            f"Failed to export {target} model for project {project_id}: {err_output}"
        )

    return inference_dir, checkpoint_prefix, run


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

        categories = []
        if image.project.type == "ocr_kie":
            categories = self._resolve_kie_categories(request, image)

        saved = self._upsert_annotations(image, recognized_payload)
        response_payload = {"shapes": saved}
        if categories:
            response_payload["categories"] = categories
        return Response(response_payload, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"])
    def classify_kie(self, request, pk=None):
        """
        Run SER/KIE classification using PaddleOCR.
        Requires prior detection and recognition results.
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

        checkpoint_type_raw = (
            request.data.get("checkpoint_type")
            or request.query_params.get("checkpoint_type")
            or "best"
        )
        checkpoint_type = str(checkpoint_type_raw).lower()
        if checkpoint_type not in ("best", "latest"):
            checkpoint_type = "best"

        model_name = (
            request.data.get("model_name")
            or request.query_params.get("model_name")
            or ACTIVE_KIE_MODEL_NAME
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

        try:
            resources = self._prepare_kie_resources(
                project_id=image.project.id,
                model_name=model_name,
                checkpoint_type=checkpoint_type,
            )
            classified_shapes, categories = self._run_kie_inference(
                image_path=image.image.path,
                shapes=shapes,
                resources=resources,
                use_gpu=use_gpu,
                max_len_per_part=max_len_per_part,
                min_overlap=min_overlap,
            )
        except FileNotFoundError as exc:
            log.exception("KIE classification missing resource: %s", exc)
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError as exc:
            log.exception("KIE classification validation error: %s", exc)
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - runtime dependency
            log.exception("KIE classification failed for image %s: %s", image.id, exc)
            return Response(
                {"error": "KIE classification failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        saved = self._upsert_annotations(image, classified_shapes)
        return Response(
            {"shapes": saved, "categories": categories},
            status=status.HTTP_200_OK,
        )

    def _prepare_kie_resources(
        self, project_id: int | str, model_name: str | None, checkpoint_type: str
    ) -> dict:
        run_id: str | None = None
        if model_name:
            prefix = _TRAINED_MODEL_KEY.format(project_id=project_id, target="kie")
            if model_name.startswith(prefix):
                suffix = model_name.replace(prefix, "", 1).lstrip("-")
                if suffix and suffix != "latest":
                    run_id = suffix
            else:
                run_id = model_name

        inference_dir, checkpoint_prefix, _ = _export_trained_model(
            project_id=project_id,
            target="kie",
            run_id=run_id,
            checkpoint_type=checkpoint_type,
        )
        checkpoint_dir = (
            checkpoint_prefix
            if checkpoint_prefix.is_dir()
            else checkpoint_prefix.parent
        )

        # Locate config.yml near the model directory (common Paddle layout).
        candidates = [
            checkpoint_dir / "config.yml",
            checkpoint_dir.parent / "config.yml",
            checkpoint_prefix.parent.parent / "config.yml",
        ]
        config_path = next((p for p in candidates if p.exists()), None)
        if not config_path:
            raise FileNotFoundError("Config file not found for trained kie model.")

        # Prefer class_list from the training dataset location if present.
        class_path = (
            checkpoint_prefix.parent.parent.parent.parent.parent
            / "datasets/kie/train_data/class_list.txt"
        )
        if not class_path.exists():
            raise FileNotFoundError(f"Class list not found at {class_path}.")

        font_path = (PADDLE_ROOT / "doc/fonts/simfang.ttf").resolve()
        if not font_path.exists():
            raise FileNotFoundError(f"Font path not found at {font_path}.")

        categories = [
            line.strip()
            for line in class_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not categories:
            raise FileNotFoundError(
                f"No categories found in class list file {class_path}."
            )

        return {
            "config_path": config_path,
            "checkpoint_prefix": checkpoint_prefix,
            "checkpoint_dir": checkpoint_dir,
            "class_path": class_path,
            "font_path": font_path,
            "categories": categories,
            "inference_dir": inference_dir,
        }

    def _canonicalize_kie_category(
        self, value, categories: list[str] | None
    ) -> str | None:
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if not cleaned or not categories:
            return cleaned
        lookup = {c.lower(): c for c in categories if isinstance(c, str)}
        return lookup.get(cleaned.lower(), cleaned)

    def _run_kie_inference(
        self,
        image_path: str,
        shapes: list[dict],
        resources: dict,
        use_gpu: bool,
        max_len_per_part: int,
        min_overlap: int,
    ) -> tuple[list[dict], list[str]]:
        img_path = Path(image_path).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}.")

        annotations, _ = self._build_kie_annotations(shapes)
        annotations = self._sort_kie_annotations(annotations)

        tokenizer = LayoutLMv2Tokenizer.from_pretrained("layoutlmv2-base-uncased")
        parts = self._split_by_token_budget(
            annotations,
            tokenizer,
            max_seq_len=max_len_per_part,
            min_token_overlap=min_overlap,
        )
        if not parts:
            parts = [annotations]
        lines = [
            f"{img_path.as_posix()}\t{json.dumps(part, ensure_ascii=False)}"
            for part in parts
            if part
        ]
        if not lines:
            raise ValueError("No valid annotations available for KIE classification.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            label_file = tmp_root / "kie_input.txt"
            label_file.write_text("\n".join(lines), encoding="utf-8")
            save_dir = tmp_root / "kie_output"
            save_dir.mkdir(parents=True, exist_ok=True)

            argv_backup = list(sys.argv)
            sys.argv = [
                "infer_kie_token_ser.py",
                "-c",
                str(resources["config_path"]),
                "-o",
                f"Global.use_gpu={use_gpu}",
                f"Global.class_path={resources['class_path'].as_posix()}",
                f"PostProcess.class_path={resources['class_path'].as_posix()}",
                f"Global.font_path={resources['font_path'].as_posix()}",
                f"Global.infer_img={label_file.as_posix()}",
                f"Global.save_res_path={save_dir.as_posix()}",
                "Global.infer_mode=False",
                f"Architecture.Backbone.checkpoints={resources['checkpoint_dir'].as_posix()}",
                "Eval.dataset.data_dir=/",
                f"Eval.dataset.label_file_list={label_file.as_posix()}",
            ]
            try:
                config, device, logger, vdl_writer = program.preprocess()
                infer_kie_token_ser.main(config, device, logger, vdl_writer)
            finally:
                sys.argv = argv_backup

            result_path = save_dir / "infer_results.txt"
            if not result_path.exists():
                raise RuntimeError("KIE inference did not produce results.")
            predictions = self._parse_kie_results(result_path)

        classified = []
        missing = []
        for idx, shape in enumerate(shapes):
            norm_points = self._normalize_kie_points(shape.get("points"))
            if not norm_points:
                raise ValueError("Invalid polygon points found in KIE payload.")
            key = self._points_key(norm_points)
            pred = predictions.get(key)
            if not pred:
                missing.append(idx)
                continue
            label = pred["label"]
            canonical_label = self._canonicalize_kie_category(
                label, resources.get("categories")
            )
            classified.append({**shape, "category": canonical_label})

        if missing:
            raise ValueError(
                f"Missing KIE predictions for {len(missing)} shapes; classification aborted."
            )

        return classified, resources["categories"]

    @staticmethod
    def _points_key(points: list[list[int]]) -> str:
        return json.dumps([[int(p[0]), int(p[1])] for p in points], ensure_ascii=False)

    def _normalize_kie_points(self, raw_points) -> list[list[int]] | None:
        pts: list[list[int]] = []
        for pt in raw_points or []:
            try:
                x, y = pt.get("x"), pt.get("y")
            except AttributeError:
                continue
            try:
                pts.append([int(round(float(x))), int(round(float(y)))])
            except (TypeError, ValueError):
                continue

        if len(pts) < 4:
            bbox = self._bbox_from_points(raw_points)
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                pts = [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                ]

        if len(pts) != 4:
            return None
        return [[int(p[0]), int(p[1])] for p in pts]

    def _normalize_prediction_points(self, raw_points) -> list[list[int]] | None:
        pts: list[list[int]] = []
        for pt in raw_points or []:
            if isinstance(pt, dict):
                x, y = pt.get("x"), pt.get("y")
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = pt[0], pt[1]
            else:
                continue
            try:
                pts.append([int(round(float(x))), int(round(float(y)))])
            except (TypeError, ValueError):
                continue
        if len(pts) != 4:
            return None
        return pts

    def _build_kie_annotations(
        self, shapes: list[dict]
    ) -> tuple[list[dict], dict[str, list[int]]]:
        annotations: list[dict] = []
        point_map: dict[str, list[int]] = {}
        for idx, shape in enumerate(shapes):
            norm_points = self._normalize_kie_points(shape.get("points"))
            if not norm_points:
                raise ValueError("Invalid polygon points found in KIE payload.")
            text = (shape.get("text") or "").strip()
            ann = {
                "transcription": text,
                "points": norm_points,
                "label": "None",
            }
            annotations.append(ann)
            key = self._points_key(norm_points)
            point_map.setdefault(key, []).append(idx)
        return annotations, point_map

    def _parse_kie_results(self, result_path: Path) -> dict[str, dict]:
        predictions: dict[str, dict] = {}
        lines = result_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                _, payload = line.split("\t", 1)
            except ValueError:
                continue
            try:
                data = json.loads(payload)
            except Exception:
                continue
            for ann in data.get("ocr_info") or []:
                norm_points = self._normalize_prediction_points(ann.get("points"))
                if not norm_points:
                    continue
                key = self._points_key(norm_points)
                label = ann.get("pred") or ann.get("label") or ann.get("key_cls")
                if not label:
                    continue
                try:
                    score = float(ann.get("score", 0) or 0)
                except (TypeError, ValueError):
                    score = 0.0
                existing = predictions.get(key)
                if existing is None or score > existing.get("score", 0):
                    predictions[key] = {"label": label, "score": score}
        return predictions

    @staticmethod
    def _sort_kie_annotations(
        annotations: list[dict], reverse: bool = False
    ) -> list[dict]:
        def _key(ann: dict):
            try:
                return ann["points"][0][1]
            except Exception:
                return 0

        return sorted(annotations, key=_key, reverse=reverse)

    @staticmethod
    def _kie_token_len(text: str, tokenizer) -> int:
        if not text:
            return 0
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def _split_by_token_budget(
        self,
        annotations: list[dict],
        tokenizer,
        max_seq_len: int,
        min_token_overlap: int = 0,
    ) -> list[list[dict]]:
        if max_seq_len <= 2:
            raise ValueError("max_seq_len must be > 2")
        if min_token_overlap < 0:
            raise ValueError("min_token_overlap must be >= 0")

        tok_counts = [
            self._kie_token_len(ann.get("transcription", ""), tokenizer)
            for ann in annotations
        ]
        n = len(annotations)
        if n == 0:
            return []

        budget = max_seq_len - 2
        prefix = [0]
        for t in tok_counts:
            prefix.append(prefix[-1] + t)

        def range_tokens(i: int, j: int):
            return prefix[j] - prefix[i]

        out: list[list[dict]] = []
        start = 0
        while start < n:
            end = start
            while end < n and range_tokens(start, end + 1) <= budget:
                end += 1
            if end == start:
                end = min(start + 1, n)

            out.append(annotations[start:end])

            if end >= n:
                break

            if min_token_overlap == 0:
                start = end
            else:
                start_prime = start
                while (
                    start_prime < end
                    and range_tokens(start_prime, end) >= min_token_overlap
                ):
                    start_prime += 1

                if start_prime == start:
                    start = end
                else:
                    start = max(start_prime - 1, start + 1)
        return out

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

    def _resolve_kie_categories(self, request, image: ImageModel) -> list[str]:
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

    def _classify_kie_shapes(
        self, shapes: list[dict], categories: list[str]
    ) -> list[dict]:
        if not shapes:
            return []
        if not categories:
            categories = DEFAULT_KIE_CATEGORIES
        classified = []
        for idx, shape in enumerate(shapes):
            category = shape.get("category") or categories[idx % len(categories)]
            canonical = self._canonicalize_kie_category(category, categories)
            classified.append({**shape, "category": canonical})
        return classified

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

    @staticmethod
    def _color_for_name(offset: int = 0) -> str:
        """
        Generate a stable, well-spaced RGBA color for a given name using a golden-ratio hue walk.
        """

        GOLDEN_RATIO = 0.61803398875

        def hsv_to_rgb(h: float, s: float, v: float):
            i = math.floor(h * 6)
            f = h * 6 - i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            mod = i % 6

            r = [v, q, p, p, t, v][mod]
            g = [t, v, v, q, p, p][mod]
            b = [p, p, t, v, v, q][mod]

            return (
                round(r * 255),
                round(g * 255),
                round(b * 255),
            )

        GOLDEN_RATIO = 0.61803398875
        hue = (offset * GOLDEN_RATIO) % 1
        r, g, b = hsv_to_rgb(hue, 0.65, 0.95)
        return f"rgba({r},{g},{b},0.6)"

    @action(detail=False, methods=["post"])
    def upload_dataset(self, request):
        """
        Bulk upload OCR annotations from a PaddleOCR-style label txt.
        Each line should be: <image_name>\\t<json annotations>.
        """
        project_id = request.data.get("project_id")
        if not project_id:
            return Response(
                {"error": "project_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        dataset_file = request.FILES.get("dataset") or request.FILES.get("file")
        raw_text = ""
        if dataset_file:
            try:
                raw_text = dataset_file.read().decode("utf-8-sig")
            except Exception:
                return Response(
                    {"error": "Unable to read dataset file."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            raw_text = request.data.get("dataset_text") or ""

        if not raw_text.strip():
            return Response(
                {"error": "No dataset content provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return Response(
                {"error": "Project not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if (
            self.allowed_project_types
            and project.type not in self.allowed_project_types
        ):
            return Response(
                {
                    "error": f"Endpoint only supports {self.allowed_project_types} projects."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        images = (
            ImageModel.objects.filter(project=project)
            .prefetch_related("ocr_annotations")
            .all()
        )
        image_lookup: dict[str, ImageModel] = {}
        for img in images:
            names = {Path(img.image.name).name}
            if img.original_filename:
                names.add(Path(img.original_filename).name)
            for name in names:
                image_lookup.setdefault(name, img)

        summary = {
            "processed_lines": 0,
            "updated_images": 0,
            "annotations": 0,
            "missing_images": [],
            "invalid_lines": 0,
        }
        updated_image_ids: set[int] = set()
        category_names: set[str] = set()
        total_lines = len([ln for ln in raw_text.splitlines() if ln.strip()])

        def _normalize_points(raw_points):
            normalized = []
            for pt in raw_points or []:
                if isinstance(pt, dict):
                    x, y = pt.get("x"), pt.get("y")
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x, y = pt[0], pt[1]
                else:
                    continue
                try:
                    normalized.append({"x": int(float(x)), "y": int(float(y))})
                except (TypeError, ValueError):
                    continue
            return normalized

        lines = raw_text.splitlines()
        progress_key = (request.user.id, project.id)
        _DATASET_PROGRESS[progress_key] = {
            "status": "running",
            "percent": 0,
            "processed": 0,
            "total": total_lines,
        }
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                img_name, ann_json = line.split("\t", 1)
            except ValueError:
                summary["invalid_lines"] += 1
                continue
            filename = Path(img_name).name
            image = image_lookup.get(filename)
            if not image:
                summary["missing_images"].append(filename)
                continue
            try:
                anns = json.loads(ann_json)
                if not isinstance(anns, list):
                    raise ValueError("Annotations must be a list.")
            except Exception:
                summary["invalid_lines"] += 1
                continue

            shapes_payload = []
            for ann in anns:
                points = _normalize_points(ann.get("points") or [])
                if not points:
                    continue
                rect_points = self._polygon_to_rect(points, tolerance_ratio=0)
                category = (ann.get("key_cls") or ann.get("category") or "").strip()
                if category:
                    category_names.add(category)
                shapes_payload.append(
                    {
                        "type": "rect" if rect_points else "polygon",
                        "points": rect_points or points,
                        "text": ann.get("transcription") or "",
                        "category": category,
                    }
                )

            saved = self._replace_annotations(image, shapes_payload)
            summary["processed_lines"] += 1
            summary["annotations"] += len(saved)
            if image.id not in updated_image_ids:
                updated_image_ids.add(image.id)
                summary["updated_images"] += 1
            if total_lines > 0:
                _DATASET_PROGRESS[progress_key] = {
                    "status": "running",
                    "percent": min(
                        100, int(summary["processed_lines"] / total_lines * 100)
                    ),
                    "processed": summary["processed_lines"],
                    "total": total_lines,
                }

        if project.type == "ocr_kie" and category_names:
            existing = {
                c.name.lower(): c for c in MaskCategory.objects.filter(project=project)
            }
            to_create = []
            for idx, name in enumerate(sorted(category_names)):
                norm = name.strip()
                if not norm:
                    continue
                if norm.lower() in existing:
                    continue
                color = self._color_for_name(offset=len(existing) + idx)
                to_create.append(MaskCategory(project=project, name=norm, color=color))
            if to_create:
                MaskCategory.objects.bulk_create(to_create)

        _DATASET_PROGRESS[progress_key] = {
            "status": "completed",
            "percent": 100,
            "processed": summary["processed_lines"],
            "total": total_lines,
        }

        return Response(
            {
                "project_id": project.id,
                "summary": summary,
                "updated_image_ids": sorted(updated_image_ids),
            },
            status=status.HTTP_200_OK,
        )

    @action(detail=False, methods=["get"])
    def dataset_progress(self, request):
        """
        Return the last known dataset import progress for a project/user.
        """
        project_id = request.query_params.get("project_id")
        if not project_id:
            return Response(
                {"error": "project_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return Response(
                {"error": "Project not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        key = (request.user.id, project.id)
        if request.query_params.get("reset"):
            _DATASET_PROGRESS[key] = {
                "status": "idle",
                "percent": 0,
                "processed": 0,
                "total": 0,
            }
        progress = _DATASET_PROGRESS.get(key) or {
            "status": "idle",
            "percent": 0,
            "processed": 0,
            "total": 0,
        }
        print("requests", progress)
        return Response({"project_id": project.id, "progress": progress})

    @action(detail=False, methods=["post"])
    def configure_trained_models(self, request):
        """
        Export and load finetuned OCR models for a project, then set them active.
        """
        global ACTIVE_DET_MODEL_NAME, ACTIVE_REC_MODEL_NAME, ACTIVE_KIE_MODEL_NAME

        project_id = request.data.get("project_id")
        if not project_id:
            return Response(
                {"error": "project_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return Response(
                {"error": "Project not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if (
            self.allowed_project_types
            and project.type not in self.allowed_project_types
        ):
            return Response(
                {
                    "error": f"Endpoint only supports {self.allowed_project_types} projects."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        requested_models = request.data.get("models") or ["det", "rec"]
        if not isinstance(requested_models, (list, tuple, set)):
            requested_models = [requested_models]
        requested_models = [m for m in requested_models if m in ("det", "rec", "kie")]
        if not requested_models:
            return Response(
                {"error": "No valid models provided. Use any of: det, rec, kie."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        runs_map = request.data.get("runs") or {}
        checkpoint_map = request.data.get("checkpoint_type") or {}

        loaded: dict[str, dict] = {}
        errors: dict[str, str] = {}

        for target in requested_models:
            if target == "det" and TextDetection is None:
                errors[target] = (
                    "PaddleOCR TextDetection unavailable. Is the submodule installed?"
                )
                continue
            if target == "rec" and TextRecognition is None:
                errors[target] = (
                    "PaddleOCR TextRecognition unavailable. Is the submodule installed?"
                )
                continue
            run_id = runs_map.get(target) if isinstance(runs_map, dict) else None
            checkpoint_type = (
                checkpoint_map.get(target) if isinstance(checkpoint_map, dict) else None
            )
            checkpoint_type = (checkpoint_type or "best").lower()
            if checkpoint_type not in ("best", "latest"):
                checkpoint_type = "best"
            try:
                inference_dir, checkpoint_prefix, run_used = _export_trained_model(
                    project.id,
                    target,
                    run_id=run_id,
                    checkpoint_type=checkpoint_type,
                )
                model_key = _TRAINED_MODEL_KEY.format(
                    project_id=project.id,
                    target=f"{target}-{run_used.id if run_used else 'latest'}",
                )
                if target == "det":
                    detector = TextDetection(model_dir=str(inference_dir))
                    _DETECTOR_CACHE[model_key] = detector
                    ACTIVE_DET_MODEL_NAME = model_key
                elif target == "rec":
                    recognizer = TextRecognition(model_dir=str(inference_dir))
                    _RECOGNIZER_CACHE[model_key] = recognizer
                    ACTIVE_REC_MODEL_NAME = model_key
                else:
                    ACTIVE_KIE_MODEL_NAME = model_key

                loaded[target] = {
                    "model_key": model_key,
                    "checkpoint": str(checkpoint_prefix),
                    "inference_dir": str(inference_dir),
                    "run_id": str(run_used.id) if run_used else None,
                    "checkpoint_type": checkpoint_type,
                }
            except FileNotFoundError as exc:
                errors[target] = str(exc)
            except Exception as exc:  # pragma: no cover - runtime dependency
                log.exception(
                    "Failed to load trained %s model for project %s: %s",
                    target,
                    project.id,
                    exc,
                )
                errors[target] = f"Failed to load {target} model: {exc}"

        if not loaded:
            return Response(
                {"error": "Unable to load trained models.", "details": errors},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "project_id": project.id,
                "loaded": loaded,
                "errors": errors or None,
                "active": {
                    "detect_model": ACTIVE_DET_MODEL_NAME,
                    "recognize_model": ACTIVE_REC_MODEL_NAME,
                    "classify_model": ACTIVE_KIE_MODEL_NAME,
                },
            },
            status=status.HTTP_200_OK,
        )

    @action(detail=False, methods=["post"])
    def configure_models(self, request):
        """
        Load and set active detect/recognize/classify models. Unload previous active models if changed.
        """
        global ACTIVE_DET_MODEL_NAME, ACTIVE_REC_MODEL_NAME, ACTIVE_KIE_MODEL_NAME

        detect_model = request.data.get("detect_model") or ACTIVE_DET_MODEL_NAME
        recognize_model = request.data.get("recognize_model") or ACTIVE_REC_MODEL_NAME
        classify_model = request.data.get("classify_model") or ACTIVE_KIE_MODEL_NAME

        changed = {"detect": False, "recognize": False, "classify": False}

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

        if classify_model and classify_model != ACTIVE_KIE_MODEL_NAME:
            ACTIVE_KIE_MODEL_NAME = classify_model
            changed["classify"] = True

        return Response(
            {
                "detect_model": ACTIVE_DET_MODEL_NAME,
                "recognize_model": ACTIVE_REC_MODEL_NAME,
                "classify_model": ACTIVE_KIE_MODEL_NAME,
                "changed": changed,
            },
            status=status.HTTP_200_OK,
        )
