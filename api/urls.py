from django.urls import include, path
from rest_framework.routers import DefaultRouter
from api.views import (
    MaskCategoryViewSet,
    ModelManagerViewSet,
    OcrImageViewSet,
    OcrTrainingDatasetView,
    OcrTrainingDefaultsView,
    OcrTrainingJobListView,
    OcrTrainingJobLogsView,
    OcrTrainingJobStopView,
    OcrTrainingJobView,
    OcrTrainingStartView,
    OcrTrainingRunListView,
    ProjectViewSet,
    SegmentationImageViewSet,
    VideoViewSet,
)

router = DefaultRouter()
router.register(r"projects", ProjectViewSet, basename="project")
router.register(r"images", SegmentationImageViewSet, basename="images")
router.register(r"ocr-images", OcrImageViewSet, basename="ocr-images")
router.register(r"video", VideoViewSet, basename="video")
router.register(r"model", ModelManagerViewSet, basename="model")
router.register(r"categories", MaskCategoryViewSet, basename="categories")

urlpatterns = [
    path("", include(router.urls)),
    path("ocr-training/defaults/", OcrTrainingDefaultsView.as_view(), name="ocr-training-defaults"),
    path("ocr-training/dataset/", OcrTrainingDatasetView.as_view(), name="ocr-training-dataset"),
    path("ocr-training/start/", OcrTrainingStartView.as_view(), name="ocr-training-start"),
    path("ocr-training/jobs/<str:job_id>/", OcrTrainingJobView.as_view(), name="ocr-training-job"),
    path("ocr-training/jobs/", OcrTrainingJobListView.as_view(), name="ocr-training-jobs"),
    path("ocr-training/jobs/<str:job_id>/stop/", OcrTrainingJobStopView.as_view(), name="ocr-training-job-stop"),
    path("ocr-training/jobs/<str:job_id>/logs/", OcrTrainingJobLogsView.as_view(), name="ocr-training-job-logs"),
    path("ocr-training/runs/", OcrTrainingRunListView.as_view(), name="ocr-training-runs"),
]
