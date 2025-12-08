from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.image import ImageModel
from api.models.project import Project
from api.serializers import ImageModelSerializer


class BaseImageViewSet(viewsets.ModelViewSet):
    """
    Shared create/list logic for image-backed projects.
    Child classes should set `allowed_project_types`.
    """

    serializer_class = ImageModelSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    allowed_project_types: tuple[str, ...] = ()

    def get_queryset(self):
        qs = ImageModel.objects.filter(project__user=self.request.user)
        if self.allowed_project_types:
            qs = qs.filter(project__type__in=self.allowed_project_types)
        return qs

    def create(self, request, *args, **kwargs):
        project_id = request.data.get("project_id")
        is_label = request.data.get("is_label", False)

        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return Response(
                {
                    "error": "Project not found or you do not have permission to access it."
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        if self.allowed_project_types and project.type not in self.allowed_project_types:
            return Response(
                {"error": f"Project type '{project.type}' is not supported by this endpoint."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        images = request.FILES.getlist("images")
        image_records = []

        for image in images:
            image_record = ImageModel.objects.create(
                image=image,
                is_label=is_label,
                project=project,
                original_filename=image.name,
            )
            image_records.append(image_record)

        serializer = self.get_serializer(
            image_records, many=True, context={"request": request}
        )
        return Response(serializer.data, status=status.HTTP_201_CREATED)
