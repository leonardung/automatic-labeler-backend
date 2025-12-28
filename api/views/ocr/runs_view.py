import shutil
from pathlib import Path

from django.http import JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.project import Project
from api.models.training import TrainingRun


def _detect_checkpoints(models_dir: Path) -> tuple[str, str]:
    """
    Best-effort checkpoint discovery on disk so Saved Models shows available weights
    even if the DB fields were not populated (e.g., mid-run or after manual copy).
    Returns (best_checkpoint, latest_checkpoint) as strings (may be empty).
    """
    if not models_dir.exists():
        return "", ""

    def _first_existing(prefixes):
        for prefix in prefixes:
            if prefix.with_suffix(".pdparams").exists():
                return str(prefix)
        for pdparams in sorted(models_dir.rglob("*.pdparams")):
            return str(pdparams.with_suffix(""))
        return ""

    best_prefix = _first_existing(
        [
            models_dir / "best_accuracy",
            models_dir / "best_model" / "model",
        ]
    )
    latest_prefix = _first_existing(
        [
            models_dir / "latest",
            models_dir / "latest" / "model",
            models_dir / "latest" / "model_state",
        ]
    )
    return best_prefix, latest_prefix


def _serialize_run(run: TrainingRun) -> dict:
    models_dir = Path(run.models_dir) if run.models_dir else None
    best_checkpoint = run.best_checkpoint
    latest_checkpoint = run.latest_checkpoint
    if models_dir and (not best_checkpoint or not latest_checkpoint):
        detected_best, detected_latest = _detect_checkpoints(models_dir)
        best_checkpoint = best_checkpoint or detected_best
        latest_checkpoint = latest_checkpoint or detected_latest
    return {
        "id": str(run.id),
        "job_id": run.job_id,
        "project_id": run.project_id,
        "target": run.target,
        "status": run.status,
        "models_dir": run.models_dir,
        "best_checkpoint": best_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "best_metric": run.best_metric or {},
        "metrics_log": run.metrics_log or [],
        "log_path": run.log_path,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


class OcrTrainingRunListView(APIView):
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

        target = request.query_params.get("target")
        status_filter = request.query_params.get("status")

        runs_qs = TrainingRun.objects.filter(project=project)
        if target in ("det", "rec", "kie"):
            runs_qs = runs_qs.filter(target=target)
        if status_filter:
            runs_qs = runs_qs.filter(status=status_filter)

        runs = [_serialize_run(run) for run in runs_qs.order_by("-created_at")[:200]]
        return JsonResponse({"runs": runs}, status=status.HTTP_200_OK)

    def delete(self, request):
        project_id = request.data.get("project_id") or request.query_params.get("project_id")
        run_id = request.data.get("run_id") or request.query_params.get("run_id")
        if not project_id or not run_id:
            return JsonResponse(
                {"error": "project_id and run_id are required."},
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
            run = TrainingRun.objects.get(id=run_id, project=project)
        except TrainingRun.DoesNotExist:
            return JsonResponse(
                {"error": "Run not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        models_dir = Path(run.models_dir) if run.models_dir else None
        run.delete()
        if models_dir and models_dir.exists():
            try:
                shutil.rmtree(models_dir)
            except Exception:
                # Best-effort cleanup; ignore filesystem errors
                pass

        return JsonResponse({"deleted": str(run_id)}, status=status.HTTP_200_OK)
