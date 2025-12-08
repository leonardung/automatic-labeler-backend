from django.db import models


class OcrAnnotation(models.Model):
    SHAPE_TYPE_CHOICES = [
        ("rect", "Rectangle"),
        ("polygon", "Polygon"),
    ]

    image = models.ForeignKey(
        "ImageModel", on_delete=models.CASCADE, related_name="ocr_annotations"
    )
    shape_type = models.CharField(max_length=16, choices=SHAPE_TYPE_CHOICES)
    points = models.JSONField(default=list)
    text = models.TextField(blank=True, default="")
    category = models.CharField(max_length=128, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"OCR {self.shape_type} #{self.pk} for image {self.image_id}"
