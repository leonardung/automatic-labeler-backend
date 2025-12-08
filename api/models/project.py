from django.db import models
from django.contrib.auth.models import User


class Project(models.Model):
    PROJECT_TYPE_CHOICES = [
        ("segmentation", "Segmentation"),
        ("video_tracking_segmentation", "Video Tracking Segmentation"),
        ("ocr", "OCR"),
        ("ocr_kie", "OCR with KIE"),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="projects")
    name = models.CharField(max_length=255)
    type = models.CharField(max_length=30, choices=PROJECT_TYPE_CHOICES)

    def __str__(self):
        return f"{self.name} ({self.type})"
