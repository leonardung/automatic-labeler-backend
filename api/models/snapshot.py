import os
from django.db import models

from api.models.image import ImageModel
from api.models.mask import MaskCategory
from api.models.project import Project


def snapshot_mask_upload_path(instance, filename):
    return (
        f"snapshots/{instance.snapshot_image.snapshot.id}/"
        f"masks/{instance.snapshot_image.image.id}/{filename}"
    )


class ProjectSnapshot(models.Model):
    project = models.ForeignKey(Project, related_name="snapshots", on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255, blank=True, default="")

    def __str__(self):
        label = self.name or self.created_at.strftime("%Y-%m-%d %H:%M")
        return f"{label} (project {self.project_id})"


class ProjectSnapshotCategory(models.Model):
    snapshot = models.ForeignKey(
        ProjectSnapshot, related_name="categories", on_delete=models.CASCADE
    )
    name = models.CharField(max_length=255)
    color = models.CharField(max_length=32, default="#00c800")
    original_category = models.ForeignKey(
        MaskCategory, null=True, blank=True, on_delete=models.SET_NULL
    )

    def __str__(self):
        return f"{self.name} (snapshot {self.snapshot_id})"


class ProjectSnapshotImage(models.Model):
    snapshot = models.ForeignKey(
        ProjectSnapshot, related_name="images", on_delete=models.CASCADE
    )
    image = models.ForeignKey(
        ImageModel, related_name="snapshots", on_delete=models.CASCADE
    )

    def __str__(self):
        return f"Image {self.image_id} snapshot {self.snapshot_id}"


class ProjectSnapshotSegmentationMask(models.Model):
    snapshot_image = models.ForeignKey(
        ProjectSnapshotImage, related_name="segmentation_masks", on_delete=models.CASCADE
    )
    category = models.ForeignKey(
        ProjectSnapshotCategory,
        related_name="segmentation_masks",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )
    mask = models.FileField(upload_to=snapshot_mask_upload_path, blank=True, null=True)
    points = models.JSONField(default=list, blank=True)

    def filename(self):
        return os.path.basename(self.mask.name) if self.mask else ""


class ProjectSnapshotOcrAnnotation(models.Model):
    snapshot_image = models.ForeignKey(
        ProjectSnapshotImage, related_name="ocr_annotations", on_delete=models.CASCADE
    )
    shape_type = models.CharField(max_length=16)
    points = models.JSONField(default=list, blank=True)
    text = models.TextField(blank=True, default="")
    category = models.CharField(max_length=128, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"OCR annotation snapshot {self.snapshot_image.snapshot_id}"
