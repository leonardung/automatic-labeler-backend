from django.db import migrations, models
import api.models.mask
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0004_alter_coordinate_unique_together"),
    ]

    operations = [
        migrations.CreateModel(
            name="MaskCategory",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=255)),
                ("color", models.CharField(default="#00c800", max_length=16)),
                (
                    "project",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="categories",
                        to="api.project",
                    ),
                ),
            ],
            options={"unique_together": {("project", "name")}},
        ),
        migrations.CreateModel(
            name="SegmentationMask",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("mask", models.ImageField(blank=True, null=True, upload_to=api.models.mask.segmentation_mask_upload_path)),
                ("points", models.JSONField(blank=True, default=list)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "category",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="masks",
                        to="api.maskcategory",
                    ),
                ),
                (
                    "image",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="masks",
                        to="api.imagemodel",
                    ),
                ),
            ],
            options={"unique_together": {("image", "category")}},
        ),
    ]
