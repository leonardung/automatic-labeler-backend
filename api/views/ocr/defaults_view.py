import copy

from django.http import JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.project import Project
from api.models.training import TrainingConfig

BASE_DEFAULTS = {
    "use_gpu": True,
    "test_ratio": 0.3,
    "train_seed": 42,
    "split_seed": 42,
    "models": {
        "det": {
            "epoch_num": 50,
            "print_batch_step": 10,
            "eval_batch_step": 200,
        },
        "rec": {
            "epoch_num": 50,
            "print_batch_step": 10,
            "eval_batch_step": 200,
        },
        "kie": {
            "epoch_num": 50,
            "print_batch_step": 10,
            "eval_batch_step": 200,
        },
    },
}


class OcrTrainingDefaultsView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        project_id = request.query_params.get("project_id")
        project = None
        if project_id:
            try:
                project = Project.objects.get(id=project_id, user=request.user)
            except Project.DoesNotExist:
                return JsonResponse(
                    {"error": "Project not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

        defaults = copy.deepcopy(BASE_DEFAULTS)
        if project:
            saved = (
                TrainingConfig.objects.filter(user=request.user, project=project)
                .order_by("-updated_at")
                .first()
            )
            if saved:
                defaults = _merge_defaults(defaults, saved.config)

        return JsonResponse({"defaults": defaults}, status=status.HTTP_200_OK)


def _merge_defaults(base_defaults: dict, saved_config: dict) -> dict:
    merged = copy.deepcopy(base_defaults)
    if not isinstance(saved_config, dict):
        return merged

    for key in ("use_gpu", "test_ratio", "train_seed", "split_seed"):
        if key in saved_config and saved_config.get(key) is not None:
            merged[key] = saved_config.get(key)

    base_models = merged.get("models") or {}
    saved_models = saved_config.get("models") or {}
    for model_key, model_defaults in base_models.items():
        if not isinstance(model_defaults, dict):
            continue
        override = saved_models.get(model_key)
        if isinstance(override, dict):
            updated = dict(model_defaults)
            for field, value in override.items():
                if value is not None:
                    updated[field] = value
            base_models[model_key] = updated
    merged["models"] = base_models
    return merged
