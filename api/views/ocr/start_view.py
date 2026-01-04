import csv
import json
import logging
import os
import random
import subprocess
import sys
import threading
import uuid
from itertools import product
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from django.conf import settings
from django.http import JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.image import ImageModel
from api.models.project import Project
from api.models.training import TrainingConfig, TrainingRun

from . import helper

log = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    """Raised when a training run is intentionally stopped by the user."""


_MAX_METRIC_RECORDS = 2000


def _create_training_run(
    project: Project, job: helper.TrainingJob, target: str, models_dir: Path
) -> TrainingRun:
    run = TrainingRun.objects.create(
        id=uuid.uuid4(),
        job_id=job.id,
        user_id=job.user_id,
        project=project,
        target=target,
        status="pending",
        models_dir=str(models_dir),
        log_path=str(job.log_path) if job.log_path else "",
    )
    return run


def _append_metric(run: TrainingRun, metric: dict):
    metrics_log = list(run.metrics_log or [])
    metrics_log.append(metric)
    if len(metrics_log) > _MAX_METRIC_RECORDS:
        metrics_log = metrics_log[-_MAX_METRIC_RECORDS:]
    run.metrics_log = metrics_log
    run.save(update_fields=["metrics_log"])


def _round_metric_values(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        try:
            return float(f"{float(val):.4g}")
        except Exception:
            return val
    if isinstance(val, dict):
        return {k: _round_metric_values(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_round_metric_values(v) for v in val]
    return val


def _ingest_metrics_csv(run: TrainingRun, csv_path: Path, start_row: int) -> int:
    """
    Read structured metrics emitted by PaddleOCR and persist any new rows.
    Returns the new row offset so callers can resume efficiently.
    """
    if run is None:
        return start_row
    if start_row < 0:
        start_row = 0
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fin:
            rows = list(csv.DictReader(fin))
    except FileNotFoundError:
        return start_row
    except Exception as exc:
        log.exception("Failed to read metrics CSV %s: %s", csv_path, exc)
        return start_row

    if start_row >= len(rows):
        return len(rows)

    for row in rows[start_row:]:
        phase = (row.get("phase") or "").strip().lower() or "train"
        metrics_raw = row.get("metrics") or "{}"
        try:
            metrics = _round_metric_values(json.loads(metrics_raw))
            if not isinstance(metrics, dict):
                continue
        except Exception:
            continue

        epoch_val = _safe_int(row.get("epoch"))
        global_step = _safe_int(row.get("global_step"))
        timestamp = row.get("timestamp")

        if phase == "best":
            run.best_metric = metrics or {}
            run.save(update_fields=["best_metric"])
            continue

        metrics = dict(metrics or {})
        if epoch_val:
            metrics.setdefault("epoch_current", epoch_val)
        if "epoch_total" not in metrics and epoch_val:
            metrics["epoch_total"] = metrics.get("epoch_total") or epoch_val
        metrics["global_step"] = global_step
        if timestamp:
            metrics.setdefault("timestamp", timestamp)
        metrics["phase"] = phase
        _append_metric(run, metrics)

    return len(rows)


def _update_run_status(
    run: TrainingRun, status_value: str, error: Optional[str] = None
):
    run.status = status_value
    run.error = error
    run.finished_at = datetime.utcnow()
    run.save(update_fields=["status", "error", "finished_at"])


def _checkpoint_prefix(models_dir: Path, prefer_best: bool) -> Optional[Path]:
    if not models_dir.exists():
        return None
    if prefer_best:
        candidates = [
            models_dir / "best_accuracy",
            models_dir / "best_model" / "model",
            models_dir / "best_model" / "best_model",
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
            models_dir / "best_model" / "best_model",
        ]
    for cand in candidates:
        if cand.with_suffix(".pdparams").exists():
            return cand
    pdparams = sorted(models_dir.rglob("*.pdparams"))
    if pdparams:
        return pdparams[0].with_suffix("")
    return None


class OcrTrainingStartView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def post(self, request):
        payload = request.data or {}
        project_id = payload.get("project_id")
        models_requested = payload.get("models") or []
        config = _normalize_training_config(payload.get("config") or {})

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

        resume_payload = payload.get("resume") or {}
        resume_settings = _normalize_resume_payload(resume_payload, valid_targets)
        resume_checkpoints: dict[str, str] = {}
        if resume_settings:
            resume_checkpoints, resume_errors = _resolve_resume_checkpoints(
                project, resume_settings
            )
            if resume_errors:
                return JsonResponse(
                    {
                        "error": "Unable to resume from the requested checkpoint.",
                        "details": resume_errors,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        job_id = uuid.uuid4().hex
        job_log_dir = _ensure_dir(Path(settings.MEDIA_ROOT) / "logs" / "training")
        job = helper.TrainingJob(
            id=job_id,
            user_id=request.user.id,
            project_id=project.id,
            targets=valid_targets,
            log_path=job_log_dir / f"{job_id}.log",
        )
        helper.TRAINING_JOBS[job_id] = job
        job.config_used = {
            "global": _extract_global_cfg(config),
            "models": config.get("models") or {},
        }

        _persist_training_config(request.user, project, config)
        job_config = dict(config)
        if resume_checkpoints:
            job_config["resume"] = resume_checkpoints
        _enqueue_job(job, job_config)

        return JsonResponse(
            {"job": helper._serialize_job(job)}, status=status.HTTP_202_ACCEPTED
        )


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


def _normalize_training_config(config: Optional[dict]) -> dict:
    """
    Flatten the training config payload so that global keys live at the top level.
    """
    config = config or {}
    normalized = dict(config)
    global_cfg = config.get("global") or {}
    normalized.pop("global", None)
    for key, value in global_cfg.items():
        normalized.setdefault(key, value)
    models_cfg = normalized.get("models") or config.get("models") or {}
    if not isinstance(models_cfg, dict):
        models_cfg = {}
    normalized["models"] = models_cfg
    return normalized


def _normalize_resume_payload(
    resume: Any, valid_targets: Iterable[str]
) -> dict[str, dict]:
    if not isinstance(resume, dict):
        return {}
    normalized: dict[str, dict] = {}
    for target in valid_targets:
        entry = resume.get(target)
        if not entry:
            continue
        if isinstance(entry, str):
            normalized[target] = {"run_id": entry, "checkpoint_type": "latest"}
            continue
        if isinstance(entry, dict):
            run_id = entry.get("run_id") or entry.get("id") or entry.get("run")
            if not run_id:
                continue
            checkpoint_type = entry.get("checkpoint_type") or entry.get(
                "checkpoint"
            )
            normalized[target] = {
                "run_id": str(run_id),
                "checkpoint_type": checkpoint_type or "latest",
            }
    return normalized


def _resolve_resume_checkpoints(
    project: Project, resume_map: dict[str, dict]
) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    errors: list[str] = []
    for target, entry in resume_map.items():
        run_id = entry.get("run_id") if isinstance(entry, dict) else None
        if not run_id:
            errors.append(f"Missing run_id for target {target}.")
            continue
        checkpoint_type = str(entry.get("checkpoint_type") or "latest").lower()
        if checkpoint_type not in ("best", "latest"):
            checkpoint_type = "latest"
        try:
            run = TrainingRun.objects.get(id=run_id, project=project)
        except TrainingRun.DoesNotExist:
            errors.append(f"Run {run_id} not found for target {target}.")
            continue
        if run.target != target:
            errors.append(f"Run {run_id} does not match target {target}.")
            continue
        models_dir = Path(run.models_dir) if run.models_dir else None
        prefix = None
        run_checkpoint = (
            run.best_checkpoint
            if checkpoint_type == "best"
            else run.latest_checkpoint
        )
        if run_checkpoint:
            candidate = Path(run_checkpoint)
            if candidate.suffix == ".pdparams":
                candidate = candidate.with_suffix("")
            if candidate.with_suffix(".pdparams").exists():
                prefix = candidate
        if not prefix and models_dir:
            prefix = _checkpoint_prefix(
                models_dir, prefer_best=checkpoint_type == "best"
            )
        if not prefix or not prefix.with_suffix(".pdparams").exists():
            errors.append(
                f"No {checkpoint_type} checkpoint found for {target} run {run_id}."
            )
            continue
        resolved[target] = prefix.as_posix()
    return resolved, errors


def _persist_training_config(user, project: Project, config: dict):
    try:
        TrainingConfig.objects.update_or_create(
            user=user, project=project, defaults={"config": config or {}}
        )
    except Exception as exc:
        log.exception(
            "Failed to persist training config for project %s: %s", project.id, exc
        )


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


def _check_stop(job: helper.TrainingJob):
    if job.stop_requested:
        raise TrainingCancelled()


def _start_next_job_locked():
    if helper.CURRENT_JOB_ID is not None or not helper.TRAINING_QUEUE:
        return
    while helper.TRAINING_QUEUE:
        next_job_id = helper.TRAINING_QUEUE.popleft()
        job = helper.TRAINING_JOBS.get(next_job_id)
        if not job:
            continue
        if job.stop_requested:
            helper._finish_job(
                job, "stopped", error=job.error or "Stopped before start."
            )
            continue
        helper.CURRENT_JOB_ID = job.id
        job.status = "running"
        job.started_at = datetime.utcnow()
        job.thread = threading.Thread(
            target=_run_training_wrapper, kwargs={"job": job}, daemon=True
        )
        job.thread.start()
        break


def _enqueue_job(job: helper.TrainingJob, config: dict):
    job.config_raw = config or {}
    job.status = "waiting"
    job.append_log("Queued for training.\n")
    with helper.TRAINING_LOCK:
        helper.TRAINING_QUEUE.append(job.id)
        _start_next_job_locked()


def _release_and_start_next(job_id: str):
    with helper.TRAINING_LOCK:
        if helper.CURRENT_JOB_ID == job_id:
            helper.CURRENT_JOB_ID = None
        _start_next_job_locked()


def _run_training(job: helper.TrainingJob):
    try:
        try:
            project = Project.objects.get(id=job.project_id, user_id=job.user_id)
        except Project.DoesNotExist:
            helper._finish_job(job, "failed", error="Project not found for this job.")
            return

        config = job.config_raw or {}
        global_cfg = _extract_global_cfg(config)
        overrides = config.get("models") or {}
        resume_checkpoints = (
            config.get("resume") if isinstance(config.get("resume"), dict) else {}
        )
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
            helper._finish_job(job, "stopped", error="Stopped by user.")
            job.append_log("Training stopped.\n")
            return
        except Exception as exc:
            log.exception("Failed to prepare datasets: %s", exc)
            helper._finish_job(
                job, "failed", error=f"Dataset preparation failed: {exc}"
            )
            return

        success = True
        runs_by_target: dict[str, TrainingRun] = {}
        for target in job.targets:
            try:
                _check_stop(job)
                run = runs_by_target.get(target)
                if run is None:
                    models_dir = _ensure_dir(
                        helper.MEDIA_PROJECT_ROOT
                        / str(project.id)
                        / "models"
                        / job.id
                        / target
                    )
                    run = _create_training_run(project, job, target, models_dir)
                    runs_by_target[target] = run
                resume_checkpoint = (
                    resume_checkpoints.get(target) if resume_checkpoints else None
                )
                if resume_checkpoint:
                    job.append_log(
                        f"[{target}] Resuming from checkpoint {resume_checkpoint}\n"
                    )
                _train_model(
                    target,
                    project,
                    dataset_info,
                    global_cfg,
                    overrides,
                    job,
                    run=run,
                    resume_checkpoint=resume_checkpoint,
                )
            except TrainingCancelled:
                success = False
                for run in runs_by_target.values():
                    if run.status not in ("completed", "failed", "stopped"):
                        _update_run_status(run, "stopped", error="Stopped by user.")
                helper._finish_job(job, "stopped", error="Stopped by user.")
                job.append_log("Training stopped.\n")
                break
            except Exception as exc:
                log.exception("Training for %s failed: %s", target, exc)
                job.append_log(f"[{target}] failed: {exc}\n")
                job.error = str(exc)
                success = False
                run = runs_by_target.get(target)
                if run:
                    _update_run_status(run, "failed", error=str(exc))
                helper._finish_job(job, "failed", error=str(exc))
                break

        if success:
            job.append_log("Training completed.\n")
            helper._finish_job(job, "completed")
    finally:
        job.current_process = None


def _run_training_wrapper(job: helper.TrainingJob):
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
    train_count = len(base_train_samples)
    test_count = len(base_val_samples)
    train_annotations = sum(len(s["shapes"]) for s in base_train_samples)
    test_annotations = sum(len(s["shapes"]) for s in base_val_samples)

    project_root = _ensure_dir(helper.MEDIA_PROJECT_ROOT / str(project.id))
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
        "samples": train_count + test_count,
        "annotations": train_annotations + test_annotations,
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
        "train_samples": train_count,
        "test_samples": test_count,
        "train_annotations": train_annotations,
        "test_annotations": test_annotations,
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
    categories = {"None"}
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
        category = shape.get("category") or "None"
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
        "num_classes": len(categories),
    }


def _train_model(
    target: str,
    project: Project,
    dataset_info: dict,
    global_cfg: dict,
    overrides: dict,
    job: helper.TrainingJob,
    run: Optional[TrainingRun] = None,
    resume_checkpoint: Optional[str] = None,
):
    _check_stop(job)
    cfg_path = {
        "det": helper.DET_CONFIG_PATH,
        "rec": helper.REC_CONFIG_PATH,
        "kie": helper.KIE_CONFIG_PATH,
    }[target]
    if not cfg_path.exists():
        raise RuntimeError(f"Missing config for {target}: {cfg_path}")

    paddle_env = os.environ.copy()
    if not global_cfg.get("use_gpu", True):
        paddle_env["CUDA_VISIBLE_DEVICES"] = ""

    if run is None:
        models_dir = _ensure_dir(
            helper.MEDIA_PROJECT_ROOT / str(project.id) / "models" / job.id / target
        )
    else:
        models_dir = Path(run.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        run.status = "running"
        run.started_at = datetime.utcnow()
        run.save(update_fields=["status", "started_at", "models_dir"])

    metrics_csv_path = models_dir / "metrics.csv"

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
                f"Global.pretrained_model={helper.PRETRAIN_ROOT / 'PP-OCRv5_server_det_pretrained.pdparams'}",
                f"Global.save_model_dir={models_dir}",
                f"Train.dataset.data_dir={det['data_dir']}",
                f"Train.dataset.label_file_list=['{det['train_label']}']",
                f"Eval.dataset.data_dir={det['data_dir']}",
                f"Eval.dataset.label_file_list=['{det['val_label']}']",
                "Train.loader.batch_size_per_card=1",
            ]
        )
    elif target == "rec":
        rec = dataset_info["rec"]
        overrides_list.extend(
            [
                f"Global.pretrained_model={helper.PRETRAIN_ROOT / 'latin_PP-OCRv5_mobile_rec_pretrained.pdparams'}",
                f"Global.character_dict_path={helper.PADDLE_ROOT / 'ppocr' / 'utils' / 'dict' / 'ppocrv5_latin_dict.txt'}",
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
            "pretrained_model": (
                helper.PRETRAIN_ROOT / "ser_vi_layoutxlm_xfund_pretrained/best_accuracy"
            ).as_posix(),
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
                "Eval.loader.batch_size_per_card=1",
                "Train.loader.batch_size_per_card=1",
            ]
        )

    if resume_checkpoint:
        resume_path = Path(str(resume_checkpoint))
        if resume_path.suffix == ".pdparams":
            resume_path = resume_path.with_suffix("")
        checkpoint_file = resume_path.with_suffix(".pdparams")
        if not checkpoint_file.exists():
            raise RuntimeError(
                f"Resume checkpoint not found at {checkpoint_file}."
            )
        overrides_list.append(f"Global.checkpoints={resume_path.as_posix()}")

    model_overrides = overrides.get(target) or {}
    model_overrides = {**model_overrides, "save_epoch_step": 100000000}
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
        cwd=helper.PADDLE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=paddle_env,
    )
    job.current_process = process
    metrics_row_offset = 0
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
            if run:
                metrics_row_offset = _ingest_metrics_csv(
                    run, metrics_csv_path, metrics_row_offset
                )
        ret = process.wait()
        if job.stop_requested:
            raise TrainingCancelled()
        if ret != 0:
            raise RuntimeError(f"{target} training exited with code {ret}")
    finally:
        job.current_process = None
        if run:
            metrics_row_offset = _ingest_metrics_csv(
                run, metrics_csv_path, metrics_row_offset
            )

    if run:
        best_prefix = _checkpoint_prefix(models_dir, prefer_best=True)
        latest_prefix = _checkpoint_prefix(models_dir, prefer_best=False)
        run.best_checkpoint = str(best_prefix) if best_prefix else ""
        run.latest_checkpoint = str(latest_prefix) if latest_prefix else ""
        _update_run_status(run, "completed")
