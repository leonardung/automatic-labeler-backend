import logging

from django.db.models import Count
from django.http import JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.image import ImageModel
from api.models.ocr import OcrAnnotation
from api.models.project import Project

log = logging.getLogger(__name__)


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


def _dataset_snapshot(project: Project) -> dict:
    """
    Summarize dataset stats without writing any files.
    """
    total_images = ImageModel.objects.filter(project=project).count()
    annotated = OcrAnnotation.objects.filter(image__project=project)
    images_with_labels = annotated.values("image_id").distinct().count()
    category_rows = annotated.values("category").annotate(total=Count("id"))
    category_counts = []
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
