from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0005_maskcategory_segmentationmask"),
    ]

    operations = [
        migrations.AlterField(
            model_name="maskcategory",
            name="color",
            field=models.CharField(default="#00c800", max_length=32),
        ),
    ]
