from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0012_trainingrun_name"),
    ]

    operations = [
        migrations.AddField(
            model_name="project",
            name="ocr_model_config",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
