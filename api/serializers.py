from rest_framework import serializers

from api.models.coordinate import Coordinate
from api.models.image import ImageModel
from api.models.mask import MaskCategory, SegmentationMask
from api.models.project import Project
from api.models.ocr import OcrAnnotation


class CoordinateSerializer(serializers.ModelSerializer):
    image_id = serializers.SerializerMethodField()

    class Meta:
        model = Coordinate
        fields = ["id", "x", "y", "include", "image_id"]

    def get_image_id(self, obj):
        return obj.image.id


class MaskCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = MaskCategory
        fields = ["id", "name", "color"]


class SegmentationMaskSerializer(serializers.ModelSerializer):
    category = MaskCategorySerializer(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=MaskCategory.objects.all(), source="category", write_only=True
    )

    class Meta:
        model = SegmentationMask
        fields = ["id", "mask", "points", "category", "category_id"]


class OcrAnnotationSerializer(serializers.ModelSerializer):
    class Meta:
        model = OcrAnnotation
        fields = ["id", "shape_type", "points", "text", "category"]


class ImageModelSerializer(serializers.ModelSerializer):
    coordinates = CoordinateSerializer(many=True, read_only=True)
    masks = SegmentationMaskSerializer(many=True, read_only=True)
    ocr_annotations = OcrAnnotationSerializer(many=True, read_only=True)

    class Meta:
        model = ImageModel
        fields = [
            "id",
            "image",
            "thumbnail",
            "uploaded_at",
            "coordinates",
            "is_label",
            "original_filename",
            "masks",
            "ocr_annotations",
        ]


class ProjectSerializer(serializers.ModelSerializer):
    images = ImageModelSerializer(many=True, read_only=True)
    categories = MaskCategorySerializer(many=True, read_only=True)

    class Meta:
        model = Project
        fields = ["id", "name", "type", "images", "categories"]
