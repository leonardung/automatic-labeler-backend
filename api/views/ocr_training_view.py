import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml
from django.conf import settings
from django.http import JsonResponse
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


def _serialize_job(job: "TrainingJob") -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "message": job.message,
        "error": job.error,
        "targets": job.targets,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "dataset": job.dataset_info or {},
        "config": job.config_used,
    }


@dataclass
class TrainingJob:
    id: str
    user_id: int
    project_id: int
    targets: List[str]
    status: str = "pending"
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    dataset_info: Optional[dict] = None
    config_used: Optional[dict] = None
    log_tail: List[str] = field(default_factory=list)
    log_path: Optional[Path] = None

    def append_log(self, line: str, persist: bool = True):
        if line:
            self.log_tail.append(line.rstrip())
            self.log_tail = self.log_tail[-40:]
            self.message = "\n".join(self.log_tail[-10:])
        if persist and self.log_path:
            with self.log_path.open("a", encoding="utf-8") as fp:
                fp.write(line)


class OcrTrainingDefaultsView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        defaults = {
            "use_gpu": True,
            "test_ratio": 0.05,
            "train_seed": None,
            "split_seed": None,
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
                "det": _load_model_defaults(DET_CONFIG_PATH),
                "rec": _load_model_defaults(REC_CONFIG_PATH),
                "kie": _load_model_defaults(KIE_CONFIG_PATH),
            },
        }
        return JsonResponse({"defaults": defaults}, status=status.HTTP_200_OK)


class OcrTrainingJobView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, job_id: str):
        job = TRAINING_JOBS.get(job_id)
        if not job or job.user_id != request.user.id:
            return JsonResponse(
                {"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND
            )
        return JsonResponse({"job": _serialize_job(job)}, status=status.HTTP_200_OK)


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

        # Fire off background work.
        thread = threading.Thread(
            target=_run_training,
            kwargs={
                "job": job,
                "project": project,
                "config": config,
            },
            daemon=True,
        )
        thread.start()

        return JsonResponse(
            {"job": _serialize_job(job)}, status=status.HTTP_202_ACCEPTED
        )


def _run_training(job: TrainingJob, project: Project, config: dict):
    job.started_at = datetime.utcnow()
    job.status = "running"
    job.append_log(f"Starting training job {job.id} for project {project.id}\n")

    try:
        dataset_info = _prepare_datasets(project, config)
        job.dataset_info = dataset_info
        job.append_log(f"Prepared datasets at {dataset_info.get('dataset_dir', '')}\n")
    except Exception as exc:
        log.exception("Failed to prepare datasets: %s", exc)
        job.error = f"Dataset preparation failed: {exc}"
        job.status = "failed"
        job.finished_at = datetime.utcnow()
        return

    overrides = config.get("models") or {}
    global_cfg = config.get("global") or {}
    job.config_used = {"global": global_cfg, "models": overrides}

    success = True
    for target in job.targets:
        try:
            _train_model(target, project, dataset_info, global_cfg, overrides, job)
        except Exception as exc:
            log.exception("Training for %s failed: %s", target, exc)
            job.append_log(f"[{target}] failed: {exc}\n")
            job.error = str(exc)
            success = False
            break

    job.status = "completed" if success else "failed"
    job.finished_at = datetime.utcnow()
    if success:
        job.append_log("Training completed.\n")


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
        samples.append(
            {
                "image_path": image.image.path,
                "filename": Path(image.image.path).name,
                "shapes": shapes,
                "width": width,
                "height": height,
            }
        )

    if not samples:
        raise RuntimeError("No OCR annotations found for this project.")

    rng.shuffle(samples)
    split_idx = int(len(samples) * (1 - float(test_ratio)))
    if split_idx <= 0:
        split_idx = max(1, len(samples) - 1)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:] or samples[:1]

    project_root = _ensure_dir(MEDIA_PROJECT_ROOT / str(project.id))
    dataset_root = _ensure_dir(project_root / "datasets")

    det_info = _write_detection_dataset(dataset_root, train_samples, val_samples)
    rec_info = _write_recognition_dataset(dataset_root, train_samples, val_samples)
    kie_info = _write_kie_dataset(dataset_root, train_samples, val_samples)

    return {
        "label_file": str(det_info["train_label"]),
        "samples": len(samples),
        "annotations": sum(len(s["shapes"]) for s in samples),
        "dataset_dir": str(dataset_root),
        "det": det_info,
        "rec": rec_info,
        "kie": kie_info,
    }


def _write_detection_dataset(dataset_root: Path, train_samples, val_samples):
    det_root = _ensure_dir(dataset_root / "det")
    img_dir = _ensure_dir(det_root / "images")
    train_label = det_root / "train.txt"
    val_label = det_root / "val.txt"

    def _write(split_samples: List[dict], target_label: Path):
        lines = []
        for sample in split_samples:
            target_img = img_dir / sample["filename"]
            if not target_img.exists():
                shutil.copy2(sample["image_path"], target_img)
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
                f"images/{sample['filename']}\t{json.dumps(det_shapes, ensure_ascii=False)}"
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
                    rows.append(f"{crop_name}\t{_clean_text(shape['text'])}")
        label_path.write_text("\n".join(rows), encoding="utf-8")

    _write(train_samples, train_dir, train_label)
    _write(val_samples, val_dir, val_label)
    return {
        "data_dir": str(rec_root),
        "train_label": str(train_label),
        "val_label": str(val_label),
    }


def _write_kie_dataset(dataset_root: Path, train_samples, val_samples):
    kie_root = _ensure_dir(dataset_root / "kie" / "train_data" / "XFUND")
    class_list_path = kie_root / "class_list_xfun.txt"
    categories = set()
    for sample in train_samples + val_samples:
        for shape in sample["shapes"]:
            if shape["category"]:
                categories.add(shape["category"])
    if not categories:
        categories.add("others")
    class_list_path.write_text("\n".join(sorted(categories)), encoding="utf-8")

    def _write(split_samples, name: str):
        image_dir = _ensure_dir(kie_root / name / "image")
        manifest_path = (
            _ensure_dir(kie_root / name)
            / f"{'train' if name=='zh_train' else 'val'}.json"
        )
        manifest = []
        for sample in split_samples:
            target_img = image_dir / sample["filename"]
            if not target_img.exists():
                shutil.copy2(sample["image_path"], target_img)
            ocr_info = []
            for shape in sample["shapes"]:
                bbox = shape.get("bbox") or _bbox_from_points(shape["points"])
                if not bbox:
                    continue
                ocr_info.append(
                    {
                        "text": _clean_text(shape["text"]),
                        "bbox": bbox,
                        "label": shape["category"] or "others",
                    }
                )
            manifest.append(
                {
                    "id": sample["filename"],
                    "img": f"image/{sample['filename']}",
                    "width": sample["width"],
                    "height": sample["height"],
                    "ocr_info": ocr_info,
                }
            )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        return manifest_path

    train_json = _write(train_samples, "zh_train")
    val_json = _write(val_samples, "zh_val")
    return {
        "data_dir": str(kie_root),
        "train_label": str(train_json),
        "val_label": str(val_json),
        "class_path": str(class_list_path),
    }


def _train_model(
    target: str,
    project: Project,
    dataset_info: dict,
    global_cfg: dict,
    overrides: dict,
    job: TrainingJob,
):
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
        overrides_list.extend(
            [
                f"Architecture.Backbone.checkpoints={PRETRAIN_ROOT / 'ser_LayoutXLM_xfun_zh'}",
                f"Global.save_model_dir={models_dir}",
                f"Train.dataset.data_dir={kie['data_dir'] / 'zh_train'}",
                f"Train.dataset.label_file_list=['{kie['train_label']}']",
                f"Eval.dataset.data_dir={kie['data_dir'] / 'zh_val'}",
                f"Eval.dataset.label_file_list=['{kie['val_label']}']",
                f"PostProcess.class_path={kie['class_path']}",
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
    assert process.stdout is not None
    for line in process.stdout:
        job.append_log(f"[{target}] {line}", persist=True)
    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"{target} training exited with code {ret}")


TRAINING_JOBS: Dict[str, TrainingJob] = {}
