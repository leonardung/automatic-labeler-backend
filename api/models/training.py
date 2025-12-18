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
