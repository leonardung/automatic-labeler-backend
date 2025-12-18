import logging
import random
from typing import Optional

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
        test_ratio_raw = request.query_params.get("test_ratio")
        split_seed_raw = request.query_params.get("split_seed")
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse(
                {"error": "Project not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            test_ratio = float(test_ratio_raw) if test_ratio_raw is not None else 0.3
        except (TypeError, ValueError):
            test_ratio = 0.3
        test_ratio = min(max(test_ratio, 0.01), 0.99)
        try:
            split_seed = int(split_seed_raw) if split_seed_raw not in (None, "") else None
        except (TypeError, ValueError):
            split_seed = None
        try:
            dataset = _dataset_snapshot(project, test_ratio=test_ratio, split_seed=split_seed)
        except Exception as exc:
            log.exception("Failed to collect dataset snapshot: %s", exc)
            return JsonResponse(
                {"error": "Unable to load dataset summary."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return JsonResponse({"dataset": dataset}, status=status.HTTP_200_OK)


def _dataset_snapshot(
    project: Project, test_ratio: float = 0.3, split_seed: Optional[int] = None
) -> dict:
    """
    Summarize dataset stats without writing any files.
    """
    total_images = ImageModel.objects.filter(project=project).count()
    annotated = OcrAnnotation.objects.filter(image__project=project)
    per_image_counts = annotated.values("image_id").annotate(total=Count("id"))
    per_image_map = {row["image_id"]: int(row.get("total") or 0) for row in per_image_counts}
    images_with_labels = len(per_image_map)
    rng = random.Random(split_seed or 0)
    image_ids = list(per_image_map.keys())
    rng.shuffle(image_ids)
    split_idx = int(len(image_ids) * (1 - float(test_ratio))) if image_ids else 0
    if split_idx <= 0 and image_ids:
        split_idx = 1
    train_ids = image_ids[:split_idx]
    test_ids = image_ids[split_idx:] or image_ids[:1]
    train_annotations = sum(per_image_map.get(img_id, 0) for img_id in train_ids)
    test_annotations = sum(per_image_map.get(img_id, 0) for img_id in test_ids)
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
        "samples": len(train_ids) + len(test_ids),
        "annotations": train_annotations + test_annotations,
        "train_samples": len(train_ids),
        "test_samples": len(test_ids),
        "train_annotations": train_annotations,
        "test_annotations": test_annotations,
    }
