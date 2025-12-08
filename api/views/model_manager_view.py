from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication

from .segmentation_image_view import SegmentationImageViewSet


class ModelManagerViewSet(viewsets.ViewSet):
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    @action(detail=False, methods=["post"])
    def load_model(self, request):
        try:
            SegmentationImageViewSet._get_sam3_video_model()
            return Response({"message": "SAM3 models loaded"})
        except Exception as exc:
            return Response(
                {"error": f"Failed to load SAM3: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def unload_model(self, request):
        return SegmentationImageViewSet.unload_model(self, request)
