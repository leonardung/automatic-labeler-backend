from django.conf import settings
from django.http import JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from . import helper


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
                "config_path": str(helper.DET_CONFIG_PATH),
                "dataset_root": str(helper.MEDIA_PROJECT_ROOT),
                "media_root": str(settings.MEDIA_ROOT),
                "images_folder": "images",
                "dataset_folder": "datasets",
                "paddle_ocr_path": str(helper.PADDLE_ROOT),
                "pretrain_root": str(helper.PRETRAIN_ROOT),
            },
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
        return JsonResponse({"defaults": defaults}, status=status.HTTP_200_OK)
