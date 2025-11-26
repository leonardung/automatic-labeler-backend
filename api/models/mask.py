import os
from django.db import models

from api.models.image import ImageModel
from api.models.project import Project


def segmentation_mask_upload_path(instance, filename):
    return f"projects/{instance.image.project.id}/masks/{instance.category.id}/{filename}"


class MaskCategory(models.Model):
    project = models.ForeignKey(Project, related_name="categories", on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    color = models.CharField(max_length=32, default="#00c800")

    class Meta:
        unique_together = ("project", "name")

    def __str__(self):
        return f"{self.name} ({self.project_id})"


class SegmentationMask(models.Model):
    image = models.ForeignKey(ImageModel, related_name="masks", on_delete=models.CASCADE)
    category = models.ForeignKey(MaskCategory, related_name="masks", on_delete=models.CASCADE)
    mask = models.ImageField(upload_to=segmentation_mask_upload_path, blank=True, null=True)
    points = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("image", "category")

    def filename(self):
        return os.path.basename(self.mask.name) if self.mask else ""
