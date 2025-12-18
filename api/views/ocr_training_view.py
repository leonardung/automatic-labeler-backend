import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from django.conf import settings
from django.http import FileResponse, JsonResponse
from django.db.models import Count
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.image import ImageModel
from api.models.ocr import OcrAnnotation
from api.models.project import Project

log = logging.getLogger(__name__)

# Default config locations provided by the user.
DET_CONFIG_PATH = (
    Path(settings.BASE_DIR)
    / "submodules"
    / "PaddleOCR"
    / "configs"
    / "det"
    / "PP-OCRv5"
    / "PP-OCRv5_server_det.yml"
)
REC_CONFIG_PATH = (
    Path(settings.BASE_DIR)
    / "submodules"
    / "PaddleOCR"
    / "configs"
    / "rec"
    / "PP-OCRv5"
    / "multi_language"
    / "latin_PP-OCRv5_mobile_rec.yml"
)
KIE_CONFIG_PATH = (
    Path(settings.BASE_DIR)
    / "submodules"
    / "PaddleOCR"
    / "configs"
    / "kie"
    / "vi_layoutxlm"
    / "ser_vi_layoutxlm_xfund_zh.yml"
)

PADDLE_ROOT = Path(settings.BASE_DIR) / "submodules" / "PaddleOCR"
MEDIA_PROJECT_ROOT = Path(settings.MEDIA_ROOT) / "projects"
PRETRAIN_ROOT = Path(settings.MEDIA_ROOT) / "pretrain_models"
LOG_PATH_ROOT = str(Path(settings.BASE_DIR).resolve())

# Shared in-memory state for coordinating training runs.
TRAINING_JOBS: Dict[str, "TrainingJob"] = {}
TRAINING_QUEUE: deque[str] = deque()
TRAINING_LOCK = threading.Lock()
CURRENT_JOB_ID: Optional[str] = None


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_int(val, default=0):
    try:
        return int(val)
    except Exception:
        return default


def _clean_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).replace("\t", " ").replace("\n", " ").strip()


def _sanitize_log_line(line: str) -> str:
    if not line:
        return line
    return line.replace(LOG_PATH_ROOT, "...")


def _parse_epoch_progress(log_lines: List[str]) -> Optional[dict]:
    """
    Find the most recent epoch progress marker in the log tail.
    Expected pattern: "epoch: [2/50]" (case-insensitive).
    """
    pattern = re.compile(r"epoch:\s*\[(\d+)\s*/\s*(\d+)\]", re.IGNORECASE)
    for line in reversed(log_lines or []):
        match = pattern.search(line)
        if match:
            try:
                current = int(match.group(1))
                total = int(match.group(2))
                return {
                    "current": current,
                    "total": total,
                }
            except Exception:
                continue
    return None


def _expand_samples(split_samples: List[Any], target: int) -> List[Any]:
    """
    Duplicate items until we reach target count. Keeps original order and cycles.
    """
    if not split_samples:
        return []
    if len(split_samples) >= target:
        return split_samples
    output = list(split_samples)
    idx = 0
    while len(output) < target:
        output.append(split_samples[idx % len(split_samples)])
        idx += 1
    return output


def _bbox_from_points(points: Iterable[dict]):
    xs, ys = [], []
    for pt in points:
        try:
            xs.append(int(pt["x"]))
            ys.append(int(pt["y"]))
        except Exception:
            return None
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x or min_y == max_y:
        return None
    return [min_x, min_y, max_x, max_y]


def sort_annotations(
    annotations: List[Dict[str, Any]], reverse: bool = False
) -> List[Dict[str, Any]]:
    def _key(ann: Dict[str, Any]):
        try:
            return ann["points"][0][1]
        except Exception:
            return 0

    return sorted(annotations, key=_key, reverse=reverse)


def _dataset_snapshot(project: Project) -> dict:
    """
    Summarize dataset stats without writing any files.
    """
    total_images = ImageModel.objects.filter(project=project).count()
    annotated = OcrAnnotation.objects.filter(image__project=project)
    images_with_labels = annotated.values("image_id").distinct().count()
    category_rows = annotated.values("category").annotate(total=Count("id"))
    category_counts: List[dict] = []
    total_boxes = 0
    for row in category_rows:
        label = row.get("category") or "Unlabeled"
        count = int(row.get("total") or 0)
        total_boxes += count
        category_counts.append({"label": label, "count": count})
    category_counts.sort(key=lambda item: item["count"], reverse=True)
    return {
        "images": images_with_labels,
        "total_images": total_images,
        "boxes": total_boxes,
        "categories": category_counts,
        "category_total": len(category_counts),
    }


def token_len(text: str, tokenizer) -> int:
    """
    Count subword tokens for a single 'word box' text as the KIE model will see it.
    """
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def split_by_token_budget(
    annotations: List[Dict[str, Any]],
    tokenizer,
    max_seq_len: int,
    min_token_overlap: int = 0,
):
    """
    Chunk 'annotations' (already in reading order) into windows that fit the token budget.
    Each window satisfies: sum(tokens) + 2 <= max_seq_len   (# +2 for [CLS], [SEP]).
    Overlap is measured in TOKENS (not boxes) between consecutive windows.
    """

    if max_seq_len <= 2:
        raise ValueError("max_seq_len must be > 2")
    if min_token_overlap < 0:
        raise ValueError("min_token_overlap must be >= 0")

    tok_counts = [
        token_len(ann.get("transcription", ""), tokenizer) for ann in annotations
    ]
    n = len(annotations)
    if n == 0:
        return []

    budget = max_seq_len - 2  # [CLS], [SEP]
    prefix = [0]
    for t in tok_counts:
        prefix.append(prefix[-1] + t)

    def range_tokens(i, j):
        return prefix[j] - prefix[i]

    out = []
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
                # Not enough tokens to satisfy the overlap requirement; move past this window.
                start = end
            else:
                # start_prime - 1 keeps the overlap condition; ensure we still progress.
                start = max(start_prime - 1, start + 1)
    return out


def _augment_file_per_grid(
    src_file: str, dst_file: str, max_lens, min_overlaps, tokenizer
):
    """Read src_file (un-augmented), write dst_file (augmented) across all param combos."""
    if not isinstance(max_lens, (list, tuple)):
        max_lens = [max_lens]
    if not isinstance(min_overlaps, (list, tuple)):
        min_overlaps = [min_overlaps]
    if not max_lens or not min_overlaps:
        raise ValueError(
            "models.kie.max_len_per_part and models.kie.min_overlap must be provided "
            "as a scalar or list with at least one value."
        )

    with open(src_file, "r", encoding="utf-8") as fin, open(
        dst_file, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            img_path, ann_json = line.split("\t", 1)
            anns = json.loads(ann_json)
            anns = sort_annotations(anns)

            new_anns: List[Dict[str, Any]] = [
                {
                    "transcription": a.get("transcription", ""),
                    "points": a.get("points", []),
                    "label": a.get("key_cls", ""),
                }
                for a in anns
            ]

            wrote_any = False
            for ml, mo in product(max_lens, min_overlaps):
                if ml <= mo:
                    continue

                parts = split_by_token_budget(
                    new_anns,
                    tokenizer,
                    max_seq_len=int(ml),
                    min_token_overlap=int(mo),
                )
                if not parts:
                    continue
                wrote_any = True
                for part in parts:
                    fout.write(f"{img_path}\t{json.dumps(part, ensure_ascii=False)}\n")

            if not wrote_any:
                fout.write(f"{img_path}\t{json.dumps(new_anns, ensure_ascii=False)}\n")


def _load_model_defaults(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to parse %s: %s", config_path, exc)
        return {}
    global_cfg = data.get("Global", {}) or {}
    return {
        "epoch_num": global_cfg.get("epoch_num"),
        "print_batch_step": global_cfg.get("print_batch_step"),
        "save_epoch_step": global_cfg.get("save_epoch_step"),
        "eval_batch_step": global_cfg.get("eval_batch_step"),
    }


def _extract_global_cfg(config: Optional[dict]) -> dict:
    config = config or {}
    explicit_global = config.get("global") or {}
    if explicit_global:
        return explicit_global
    # Backwards compatibility: allow top-level global keys.
    return {
        key: config.get(key)
        for key in ("use_gpu", "test_ratio", "train_seed", "split_seed")
        if config.get(key) is not None
    }


def _check_stop(job: "TrainingJob"):
    if job.stop_requested:
        raise TrainingCancelled()


def _serialize_job(job: "TrainingJob", include_logs: bool = False) -> dict:
    logs_content = ""
    if include_logs and job.log_path and job.log_path.exists():
        try:
            logs_content = _sanitize_log_line(job.log_path.read_text(encoding="utf-8"))
        except Exception:
            logs_content = ""
    progress_info = _parse_epoch_progress(job.log_tail)
    progress_percent = None
    progress_label = None
    if progress_info:
        current = progress_info.get("current") or 0
        total = progress_info.get("total") or 0
        if total > 0:
            progress_percent = min(100, max(0, int(current * 100 / total)))
            progress_label = f"Epoch {current}/{total}"
        else:
            progress_label = f"Epoch {current}"
    with TRAINING_LOCK:
        queue_position: Optional[int] = None
        if job.status == "waiting":
            try:
                queue_position = list(TRAINING_QUEUE).index(job.id) + 1
            except ValueError:
                queue_position = None
        elif CURRENT_JOB_ID == job.id:
            queue_position = 0
    return {
        "id": job.id,
        "status": job.status,
        "message": job.message,
        "error": job.error,
        "targets": job.targets,
        "queue_position": queue_position,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "dataset": job.dataset_info or {},
        "config": job.config_used,
        "log_available": bool(job.log_path and job.log_path.exists()),
        "logs": logs_content if include_logs else None,
        "progress": {
            "current": progress_info.get("current") if progress_info else None,
            "total": progress_info.get("total") if progress_info else None,
            "percent": progress_percent,
            "label": progress_label,
        },
    }


@dataclass
class TrainingJob:
    id: str
    user_id: int
    project_id: int
    targets: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    dataset_info: Optional[dict] = None
    config_used: Optional[dict] = None
    log_tail: List[str] = field(default_factory=list)
    log_path: Optional[Path] = None
    config_raw: dict = field(default_factory=dict)
    stop_requested: bool = False
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    current_process: Optional[subprocess.Popen] = field(default=None, repr=False)

    def append_log(self, line: str, persist: bool = True):
        if line:
            clean_line = _sanitize_log_line(line)
            self.log_tail.append(clean_line.rstrip())
            self.log_tail = self.log_tail[-40:]
            self.message = "\n".join(self.log_tail[-10:])
        if persist and self.log_path:
            with self.log_path.open("a", encoding="utf-8") as fp:
                fp.write(_sanitize_log_line(line))


class TrainingCancelled(Exception):
    """Raised when a training run is intentionally stopped by the user."""


def _finish_job(job: "TrainingJob", status_value: str, error: Optional[str] = None):
    job.status = status_value
    job.error = error
    job.finished_at = datetime.utcnow()


def _start_next_job_locked():
    global CURRENT_JOB_ID
    if CURRENT_JOB_ID is not None or not TRAINING_QUEUE:
        return
    while TRAINING_QUEUE:
        next_job_id = TRAINING_QUEUE.popleft()
        job = TRAINING_JOBS.get(next_job_id)
        if not job:
            continue
        if job.stop_requested:
            _finish_job(job, "stopped", error=job.error or "Stopped before start.")
            continue
        CURRENT_JOB_ID = job.id
        job.status = "running"
        job.started_at = datetime.utcnow()
        job.thread = threading.Thread(
            target=_run_training_wrapper, kwargs={"job": job}, daemon=True
        )
        job.thread.start()
        break


def _enqueue_job(job: "TrainingJob", config: dict):
    job.config_raw = config or {}
    job.status = "waiting"
    job.append_log("Queued for training.\n")
    with TRAINING_LOCK:
        TRAINING_QUEUE.append(job.id)
        _start_next_job_locked()


def _release_and_start_next(job_id: str):
    with TRAINING_LOCK:
        global CURRENT_JOB_ID
        if CURRENT_JOB_ID == job_id:
            CURRENT_JOB_ID = None
        _start_next_job_locked()


def _stop_job(job: "TrainingJob"):
    if job.status in ("completed", "failed", "stopped"):
        return
    job.stop_requested = True
    job.append_log("Stop requested by user.\n")
    with TRAINING_LOCK:
        if job.status == "waiting":
            try:
                TRAINING_QUEUE.remove(job.id)
            except ValueError:
                pass
            _finish_job(job, "stopped", error="Stopped by user.")
            return
    if job.status == "running" and job.current_process:
        proc = job.current_process
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


class OcrTrainingDefaultsView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        defaults = {
            "use_gpu": True,
            "test_ratio": 0.3,
            "train_seed": 42,
            "split_seed": 42,
            "paths": {
                "config_path": str(DET_CONFIG_PATH),
                "dataset_root": str(MEDIA_PROJECT_ROOT),
                "media_root": str(settings.MEDIA_ROOT),
                "images_folder": "images",
                "dataset_folder": "datasets",
                "paddle_ocr_path": str(PADDLE_ROOT),
                "pretrain_root": str(PRETRAIN_ROOT),
            },
            "models": {
                "det": {
                    "epoch_num": 50,
                    "print_batch_step": 10,
                    "save_epoch_step": 1e111,
                    "eval_batch_step": 200,
                },
                "rec": {
                    "epoch_num": 50,
                    "print_batch_step": 10,
                    "save_epoch_step": 1e111,
                    "eval_batch_step": 200,
                },
                "kie": {
                    "epoch_num": 50,
                    "print_batch_step": 10,
                    "save_epoch_step": 1e111,
                    "eval_batch_step": 200,
                },
            },
        }
        return JsonResponse({"defaults": defaults}, status=status.HTTP_200_OK)


class OcrTrainingDatasetView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        project_id = request.query_params.get("project_id")
        if not project_id:
            return JsonResponse(
                {"error": "project_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse(
                {"error": "Project not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            dataset = _dataset_snapshot(project)
        except Exception as exc:
            log.exception("Failed to collect dataset snapshot: %s", exc)
            return JsonResponse(
                {"error": "Unable to load dataset summary."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return JsonResponse({"dataset": dataset}, status=status.HTTP_200_OK)


class OcrTrainingJobView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, job_id: str):
        job = TRAINING_JOBS.get(job_id)
        if not job or job.user_id != request.user.id:
            return JsonResponse(
                {"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND
            )
        return JsonResponse(
            {"job": _serialize_job(job, include_logs=True)},
            status=status.HTTP_200_OK,
        )


class OcrTrainingJobListView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        jobs = [job for job in TRAINING_JOBS.values() if job.user_id == request.user.id]
        jobs_sorted = sorted(jobs, key=lambda j: j.created_at, reverse=True)
        return JsonResponse(
            {"jobs": [_serialize_job(job) for job in jobs_sorted]},
            status=status.HTTP_200_OK,
        )


class OcrTrainingJobStopView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def post(self, request, job_id: str):
        job = TRAINING_JOBS.get(job_id)
        if not job or job.user_id != request.user.id:
            return JsonResponse(
                {"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND
            )
        _stop_job(job)
        return JsonResponse({"job": _serialize_job(job)}, status=status.HTTP_200_OK)


class OcrTrainingJobLogsView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, job_id: str):
        job = TRAINING_JOBS.get(job_id)
        if not job or job.user_id != request.user.id:
            return JsonResponse(
                {"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND
            )
        if not job.log_path or not job.log_path.exists():
            return JsonResponse(
                {"error": "Log not found for this job."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return FileResponse(
            job.log_path.open("rb"), as_attachment=True, filename=f"{job.id}.log"
        )


class OcrTrainingStartView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def post(self, request):
        payload = request.data or {}
        project_id = payload.get("project_id")
        models_requested = payload.get("models") or []
        config = payload.get("config") or {}

        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse(
                {"error": "Project not found."}, status=status.HTTP_404_NOT_FOUND
            )

        if project.type not in ("ocr", "ocr_kie"):
            return JsonResponse(
                {"error": "Training only supported for OCR / OCR KIE projects."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        valid_targets = [m for m in models_requested if m in ("det", "rec", "kie")]
        if not valid_targets:
            return JsonResponse(
                {"error": "No valid models specified."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        job_id = uuid.uuid4().hex
        job_log_dir = _ensure_dir(Path(settings.MEDIA_ROOT) / "logs" / "training")
        job = TrainingJob(
            id=job_id,
            user_id=request.user.id,
            project_id=project.id,
            targets=valid_targets,
            log_path=job_log_dir / f"{job_id}.log",
        )
        TRAINING_JOBS[job_id] = job
        job.config_used = {
            "global": _extract_global_cfg(config),
            "models": config.get("models") or {},
        }

        _enqueue_job(job, config)

        return JsonResponse(
            {"job": _serialize_job(job)}, status=status.HTTP_202_ACCEPTED
        )


def _run_training(job: TrainingJob):
    try:
        try:
            project = Project.objects.get(id=job.project_id, user_id=job.user_id)
        except Project.DoesNotExist:
            _finish_job(job, "failed", error="Project not found for this job.")
            return

        config = job.config_raw or {}
        global_cfg = _extract_global_cfg(config)
        overrides = config.get("models") or {}
        config_for_dataset = dict(config)
        for key, value in global_cfg.items():
            config_for_dataset.setdefault(key, value)
        job.config_used = {"global": global_cfg, "models": overrides}

        try:
            _check_stop(job)
            job.append_log(f"Starting training job {job.id} for project {project.id}\n")
            dataset_info = _prepare_datasets(project, config_for_dataset)
            job.dataset_info = dataset_info
            if "images_folder" not in global_cfg and dataset_info.get("images_dir"):
                global_cfg["images_folder"] = dataset_info["images_dir"]
            job.append_log(
                f"Prepared datasets at {dataset_info.get('dataset_dir', '')}\n"
            )
        except TrainingCancelled:
            _finish_job(job, "stopped", error="Stopped by user.")
            job.append_log("Training stopped.\n")
            return
        except Exception as exc:
            log.exception("Failed to prepare datasets: %s", exc)
            _finish_job(job, "failed", error=f"Dataset preparation failed: {exc}")
            return

        success = True
        for target in job.targets:
            try:
                _check_stop(job)
                _train_model(target, project, dataset_info, global_cfg, overrides, job)
            except TrainingCancelled:
                success = False
                _finish_job(job, "stopped", error="Stopped by user.")
                job.append_log("Training stopped.\n")
                break
            except Exception as exc:
                log.exception("Training for %s failed: %s", target, exc)
                job.append_log(f"[{target}] failed: {exc}\n")
                job.error = str(exc)
                success = False
                _finish_job(job, "failed", error=str(exc))
                break

        if success:
            job.append_log("Training completed.\n")
            _finish_job(job, "completed")
    finally:
        job.current_process = None


def _run_training_wrapper(job: TrainingJob):
    try:
        _run_training(job)
    finally:
        _release_and_start_next(job.id)


def _prepare_datasets(project: Project, config: dict) -> dict:
    test_ratio = config.get("test_ratio") or 0.05
    split_seed = config.get("split_seed")
    rng = random.Random(split_seed or 0)

    images = (
        ImageModel.objects.filter(project=project)
        .prefetch_related("ocr_annotations")
        .all()
    )

    samples = []
    total_images = images.count()
    category_counts: Dict[str, int] = {}
    total_boxes = 0
    for image in images:
        anns = list(image.ocr_annotations.all())
        if not anns:
            continue
        try:
            from PIL import Image

            with Image.open(image.image.path) as img:
                width, height = img.size
        except Exception:
            width = height = 0
        shapes = []
        for ann in anns:
            bbox = _bbox_from_points(ann.points)
            shapes.append(
                {
                    "text": _clean_text(ann.text or ""),
                    "points": ann.points,
                    "bbox": bbox,
                    "category": ann.category or "",
                }
            )
            key = ann.category or "Unlabeled"
            category_counts[key] = category_counts.get(key, 0) + 1
        samples.append(
            {
                "image_path": image.image.path,
                "filename": Path(image.image.path).name,
                "shapes": shapes,
                "width": width,
                "height": height,
            }
        )
        total_boxes += len(shapes)

    if not samples:
        raise RuntimeError("No OCR annotations found for this project.")

    rng.shuffle(samples)
    split_idx = int(len(samples) * (1 - float(test_ratio)))
    if split_idx <= 0:
        split_idx = max(1, len(samples) - 1)
    base_train_samples = samples[:split_idx]
    base_val_samples = samples[split_idx:] or samples[:1]

    # Duplicate to desired dataset sizes
    train_target = 100
    val_target = 8
    train_samples = _expand_samples(base_train_samples, train_target)
    val_samples = _expand_samples(base_val_samples, val_target)

    project_root = _ensure_dir(MEDIA_PROJECT_ROOT / str(project.id))
    dataset_root = _ensure_dir(project_root / "datasets")

    det_info = _write_detection_dataset(dataset_root, train_samples, val_samples)
    rec_info = _write_recognition_dataset(dataset_root, train_samples, val_samples)
    images_root = Path(base_train_samples[0]["image_path"]).parent
    kie_info = _write_kie_dataset(
        dataset_root,
        base_train_samples,
        base_val_samples,
        images_root=images_root,
        config=config,
        train_target=train_target,
        val_target=val_target,
    )

    return {
        "label_file": str(det_info["train_label"]),
        "samples": len(train_samples) + len(val_samples),
        "annotations": sum(len(s["shapes"]) for s in train_samples + val_samples),
        "dataset_dir": str(dataset_root),
        "images_dir": str(images_root),
        "images": len(samples),
        "total_images": total_images,
        "boxes": total_boxes,
        "categories": sorted(
            [{"label": k, "count": v} for k, v in category_counts.items()],
            key=lambda item: item["count"],
            reverse=True,
        ),
        "category_total": len(category_counts),
        "det": det_info,
        "rec": rec_info,
        "kie": kie_info,
    }


def _write_detection_dataset(dataset_root: Path, train_samples, val_samples):
    det_root = _ensure_dir(dataset_root / "det")
    train_label = det_root / "train.txt"
    val_label = det_root / "val.txt"

    def _write(split_samples: List[dict], target_label: Path):
        lines = []
        for sample in split_samples:
            rel_path = Path(os.path.relpath(sample["image_path"], start=det_root))
            det_shapes = []
            for shape in sample["shapes"]:
                points = [
                    [_safe_int(p["x"]), _safe_int(p["y"])] for p in shape["points"]
                ]
                det_shapes.append(
                    {
                        "transcription": _clean_text(shape["text"]),
                        "points": points,
                        "difficult": False,
                        "key_cls": shape["category"],
                    }
                )
            lines.append(
                f"{rel_path.as_posix()}\t{json.dumps(det_shapes, ensure_ascii=False)}"
            )
        target_label.write_text("\n".join(lines), encoding="utf-8")

    _write(train_samples, train_label)
    _write(val_samples, val_label)

    return {
        "data_dir": str(det_root),
        "train_label": str(train_label),
        "val_label": str(val_label),
    }


def _write_recognition_dataset(dataset_root: Path, train_samples, val_samples):
    from PIL import Image

    rec_root = _ensure_dir(dataset_root / "rec")
    train_dir = _ensure_dir(rec_root / "train")
    val_dir = _ensure_dir(rec_root / "val")
    train_label = rec_root / "train_list.txt"
    val_label = rec_root / "eval_list.txt"

    def _write(split_samples, out_dir: Path, label_path: Path):
        rows = []
        for sample in split_samples:
            with Image.open(sample["image_path"]) as img:
                for idx, shape in enumerate(sample["shapes"]):
                    bbox = shape.get("bbox") or _bbox_from_points(shape["points"])
                    if not bbox:
                        continue
                    min_x, min_y, max_x, max_y = bbox
                    crop = img.crop((min_x, min_y, max_x, max_y))
                    crop_name = f"{Path(sample['filename']).stem}_{idx}.png"
                    crop_path = out_dir / crop_name
                    crop.save(crop_path)
                    rows.append(
                        f"{'/'.join(crop_path.parts[-2:])}\t{_clean_text(shape['text'])}"
                    )
        label_path.write_text("\n".join(rows), encoding="utf-8")

    _write(train_samples, train_dir, train_label)
    _write(val_samples, val_dir, val_label)
    return {
        "data_dir": str(rec_root),
        "train_label": str(train_label),
        "val_label": str(val_label),
    }


def _write_kie_dataset(
    dataset_root: Path,
    train_samples,
    val_samples,
    images_root: Path,
    config: dict,
    train_target: Optional[int] = None,
    val_target: Optional[int] = None,
):
    """
    Generate SER/KIE dataset files in the expected PaddleOCR txt format:
    one sample per line: <image_path>\t<json anns>.
    """
    kie_root = _ensure_dir(dataset_root / "kie" / "train_data")
    class_list_path = kie_root / "class_list.txt"
    # Always include a catch-all class so empty/missing labels map safely.
    categories = {"others"}
    for sample in train_samples + val_samples:
        for shape in sample["shapes"]:
            if shape["category"]:
                categories.add(shape["category"])
    class_list_path.write_text("\n".join(sorted(categories)), encoding="utf-8")

    kie_cfg = (config.get("models") or {}).get("kie") or {}
    train_cfg = kie_cfg.get("train") or {}
    test_cfg = kie_cfg.get("test") or {}

    max_lens_train = train_cfg.get("max_len_per_part", [256, 512])
    min_overlaps_train = train_cfg.get(
        "min_overlap", [0, 32, 64, 96, 128, 160, 192, 224]
    )
    max_lens_test = test_cfg.get("max_len_per_part", [512])
    min_overlaps_test = test_cfg.get("min_overlap", [64, 128])

    train_raw = kie_root / "train.txt.src"
    val_raw = kie_root / "val.txt.src"
    train_txt = kie_root / "train.txt"
    val_txt = kie_root / "val.txt"

    def _shape_to_ann(shape: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_points = shape.get("points") or []
        pts: List[List[int]] = []
        for pt in raw_points:
            try:
                pts.append([_safe_int(pt.get("x")), _safe_int(pt.get("y"))])
            except Exception:
                continue
        if len(pts) < 4:
            bbox = shape.get("bbox") or _bbox_from_points(raw_points)
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                pts = [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                ]
        if len(pts) < 4:
            return None
        category = shape.get("category") or "others"
        return {
            "transcription": _clean_text(shape.get("text", "")),
            "points": pts,
            "key_cls": category,
        }

    def _write_raw(split_samples, target_path: Path):
        rows = []
        for sample in split_samples:
            rel_img = Path(os.path.relpath(sample["image_path"], start=images_root))
            anns = []
            for shape in sample["shapes"]:
                ann = _shape_to_ann(shape)
                if ann:
                    anns.append(ann)
            if not anns:
                continue
            rows.append(f"{rel_img.as_posix()}\t{json.dumps(anns, ensure_ascii=False)}")
        target_path.write_text("\n".join(rows), encoding="utf-8")

    _write_raw(train_samples, train_raw)
    _write_raw(val_samples, val_raw)

    from paddlenlp.transformers import LayoutLMv2Tokenizer

    tokenizer = LayoutLMv2Tokenizer.from_pretrained("layoutlmv2-base-uncased")

    _augment_file_per_grid(
        src_file=str(train_raw),
        dst_file=str(train_txt),
        max_lens=max_lens_train,
        min_overlaps=min_overlaps_train,
        tokenizer=tokenizer,
    )
    _augment_file_per_grid(
        src_file=str(val_raw),
        dst_file=str(val_txt),
        max_lens=max_lens_test,
        min_overlaps=min_overlaps_test,
        tokenizer=tokenizer,
    )

    def _expand_augmented_file(target_path: Path, target_count: Optional[int]):
        if not target_count or target_count <= 0:
            return
        raw_lines = target_path.read_text(encoding="utf-8").splitlines()
        lines = [line for line in raw_lines if line.strip()]
        if not lines:
            return
        expanded = _expand_samples(lines, int(target_count))
        target_path.write_text("\n".join(expanded), encoding="utf-8")

    _expand_augmented_file(train_txt, train_target)
    _expand_augmented_file(val_txt, val_target)

    for temp_file in (train_raw, val_raw):
        try:
            os.remove(temp_file)
        except OSError:
            pass

    return {
        "data_dir": str(kie_root),
        "images_dir": str(images_root),
        "train_label": str(train_txt),
        "val_label": str(val_txt),
        "class_path": str(class_list_path),
        "num_classes": len(categories) - 1,
    }


def _train_model(
    target: str,
    project: Project,
    dataset_info: dict,
    global_cfg: dict,
    overrides: dict,
    job: TrainingJob,
):
    _check_stop(job)
    cfg_path = {
        "det": DET_CONFIG_PATH,
        "rec": REC_CONFIG_PATH,
        "kie": KIE_CONFIG_PATH,
    }[target]
    if not cfg_path.exists():
        raise RuntimeError(f"Missing config for {target}: {cfg_path}")

    paddle_env = os.environ.copy()
    if not global_cfg.get("use_gpu", True):
        paddle_env["CUDA_VISIBLE_DEVICES"] = ""

    models_dir = _ensure_dir(MEDIA_PROJECT_ROOT / str(project.id) / "models" / target)

    cmd = [
        str(Path(sys.executable if "sys" in globals() else "python")),
        "tools/train.py",
        "-c",
        str(cfg_path),
    ]
    overrides_list = []

    if target == "det":
        det = dataset_info["det"]
        overrides_list.extend(
            [
                f"Global.pretrained_model={PRETRAIN_ROOT / 'PP-OCRv5_server_det_pretrained.pdparams'}",
                f"Global.save_model_dir={models_dir}",
                f"Train.dataset.data_dir={det['data_dir']}",
                f"Train.dataset.label_file_list=['{det['train_label']}']",
                f"Eval.dataset.data_dir={det['data_dir']}",
                f"Eval.dataset.label_file_list=['{det['val_label']}']",
            ]
        )
    elif target == "rec":
        rec = dataset_info["rec"]
        overrides_list.extend(
            [
                f"Global.pretrained_model={PRETRAIN_ROOT / 'latin_PP-OCRv5_mobile_rec_pretrained.pdparams'}",
                f"Global.character_dict_path={PADDLE_ROOT / 'ppocr' / 'utils' / 'dict' / 'ppocrv5_latin_dict.txt'}",
                f"Global.save_model_dir={models_dir}",
                f"Train.dataset.data_dir={rec['data_dir']}",
                f"Train.dataset.label_file_list=['{rec['train_label']}']",
                f"Eval.dataset.data_dir={rec['data_dir']}",
                f"Eval.dataset.label_file_list=['{rec['val_label']}']",
            ]
        )
    elif target == "kie":
        kie = dataset_info["kie"]
        class_count = max(1, int(kie.get("num_classes") or 0))
        # SER labels use B-*, I-* plus O -> 2 * class_count + 1.
        ser_label_classes = int(2 * class_count + 1)
        images_folder = global_cfg.get("images_folder") or kie.get("images_dir")
        if not images_folder:
            raise RuntimeError("Images folder is required for KIE training.")
        images_folder = Path(images_folder).as_posix()
        class_path = Path(kie["class_path"]).as_posix()
        train_label_path = Path(kie["train_label"]).as_posix()
        val_label_path = Path(kie["val_label"]).as_posix()
        kie_model_cfg = {
            "class_path": class_path,
            "pretrained_model": (PRETRAIN_ROOT / "ser_LayoutXLM_xfun_zh").as_posix(),
            "dataset_train": f'["{train_label_path}"]',
            "dataset_val": f'["{val_label_path}"]',
        }
        overrides_list.extend(
            [
                f"Global.save_model_dir={models_dir}",
                f"Global.class_path={kie_model_cfg['class_path']}",
                f"PostProcess.class_path={kie_model_cfg['class_path']}",
                f"Train.dataset.transforms.1.VQATokenLabelEncode.class_path={kie_model_cfg['class_path']}",
                f"Eval.dataset.transforms.1.VQATokenLabelEncode.class_path={kie_model_cfg['class_path']}",
                f"Architecture.Backbone.pretrained={kie_model_cfg['pretrained_model']}",
                f"Architecture.Backbone.num_classes={ser_label_classes}",
                f"Loss.num_classes={ser_label_classes}",
                f"Train.dataset.data_dir={images_folder}",
                f"Train.dataset.label_file_list={kie_model_cfg['dataset_train']}",
                "Train.dataset.ratio_list=1",
                f"Eval.dataset.data_dir={images_folder}",
                f"Eval.dataset.label_file_list={kie_model_cfg['dataset_val']}",
                "Eval.dataset.ratio_list=1",
            ]
        )

    model_overrides = overrides.get(target) or {}
    for key, value in model_overrides.items():
        overrides_list.append(f"Global.{key}={value}")

    if not global_cfg.get("use_gpu", True):
        overrides_list.append("Global.use_gpu=False")

    if overrides_list:
        cmd.append("-o")
        cmd.extend(overrides_list)

    job.append_log(f"[{target}] Running: {' '.join(cmd)}\n")

    process = subprocess.Popen(
        cmd,
        cwd=PADDLE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=paddle_env,
    )
    job.current_process = process
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if job.stop_requested:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TrainingCancelled()
            job.append_log(f"[{target}] {line}", persist=True)
        ret = process.wait()
        if job.stop_requested:
            raise TrainingCancelled()
        if ret != 0:
            raise RuntimeError(f"{target} training exited with code {ret}")
    finally:
        job.current_process = None
