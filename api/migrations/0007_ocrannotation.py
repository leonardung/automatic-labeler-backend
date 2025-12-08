from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0006_alter_maskcategory_color"),
    ]

    operations = [
        migrations.CreateModel(
            name="OcrAnnotation",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "shape_type",
                    models.CharField(
                        choices=[("rect", "Rectangle"), ("polygon", "Polygon")],
                        max_length=16,
                    ),
                ),
                ("points", models.JSONField(default=list)),
                ("text", models.TextField(blank=True, default="")),
                ("category", models.CharField(blank=True, max_length=128, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "image",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="ocr_annotations",
                        to="api.imagemodel",
                    ),
                ),
            ],
            options={
                "ordering": ["id"],
            },
        ),
    ]
