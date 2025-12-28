from django.conf import settings
from django.db import models


class TrainingConfig(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="training_configs"
    )
    project = models.ForeignKey(
        "api.Project", on_delete=models.CASCADE, related_name="training_configs"
    )
    config = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "project")

    def __str__(self):
        return f"TrainingConfig<{self.project_id}>"


class TrainingRun(models.Model):
    TARGET_CHOICES = [
        ("det", "Detection"),
        ("rec", "Recognition"),
        ("kie", "KIE"),
    ]
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("stopped", "Stopped"),
    ]

    id = models.UUIDField(primary_key=True, editable=False)
    job_id = models.CharField(max_length=64, db_index=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="training_runs",
    )
    project = models.ForeignKey(
        "api.Project", on_delete=models.CASCADE, related_name="training_runs"
    )
    target = models.CharField(max_length=8, choices=TARGET_CHOICES)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="pending")
    models_dir = models.CharField(max_length=512, blank=True)
    best_checkpoint = models.CharField(max_length=512, blank=True)
    latest_checkpoint = models.CharField(max_length=512, blank=True)
    best_metric = models.JSONField(default=dict, blank=True)
    metrics_log = models.JSONField(default=list, blank=True)
    log_path = models.CharField(max_length=512, blank=True)
    error = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at", "target"]

    def __str__(self):
        return f"TrainingRun<{self.target}:{self.id}>"
