import re
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from django.conf import settings

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
                return {"current": current, "total": total}
            except Exception:
                continue
    return None


@dataclass
class TrainingJob:
    id: str
    user_id: int
    project_id: int
    targets: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
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
        if persist and self.log_path:
            with self.log_path.open("a", encoding="utf-8") as fp:
                fp.write(_sanitize_log_line(line))


def _finish_job(job: "TrainingJob", status_value: str, error: Optional[str] = None):
    job.status = status_value
    job.error = error
    job.finished_at = datetime.utcnow()


def _public_dataset_info(dataset: Optional[dict]) -> dict:
    """
    Strip path-heavy fields from dataset metadata before returning to clients.
    """
    if not dataset:
        return {}
    allowed_keys = {
        "samples",
        "annotations",
        "images",
        "total_images",
        "boxes",
        "categories",
        "category_total",
    }
    public = {key: dataset.get(key) for key in allowed_keys if key in dataset}
    categories = dataset.get("categories")
    if categories:
        public["categories"] = [
            {"label": cat.get("label"), "count": cat.get("count")}
            for cat in categories
            if isinstance(cat, dict)
        ]
    return public


def _public_config(config: Optional[dict]) -> Optional[dict]:
    """
    Remove path-like fields from config metadata sent to clients.
    """
    if not config:
        return config
    global_cfg = dict(config.get("global") or {})
    global_cfg.pop("images_folder", None)
    return {
        "global": global_cfg,
        "models": config.get("models") or {},
    }


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
        "error": job.error,
        "targets": job.targets,
        "queue_position": queue_position,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "dataset": _public_dataset_info(job.dataset_info),
        "config": _public_config(job.config_used),
        "log_available": bool(job.log_path and job.log_path.exists()),
        "logs": logs_content if include_logs else None,
        "progress": {
            "current": progress_info.get("current") if progress_info else None,
            "total": progress_info.get("total") if progress_info else None,
            "percent": progress_percent,
            "label": progress_label,
        },
    }
