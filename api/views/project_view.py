from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.image import ImageModel
from api.models.mask import SegmentationMask
from api.models.project import Project
from api.serializers import ProjectSerializer


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        return Project.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["delete"])
    def delete_masks(self, request, pk=None):
        """
        Deletes the mask files for all images associated with this project.
        """
        project = self.get_object()
        masks = SegmentationMask.objects.filter(image__project=project)
        count = masks.count()
        for m in masks:
            if m.mask:
                m.mask.delete(save=False)
        masks.delete()
        return Response(
            {"detail": f"Deleted masks for {count} images in the project."},
            status=status.HTTP_200_OK,
        )

    @action(detail=True, methods=["delete"])
    def delete_coordinates(self, request, pk=None):
        """
        Deletes all coordinates associated with images in this project.
        """
        project = self.get_object()
        project_images = ImageModel.objects.filter(project=project)
        total_deleted = 0
        for image in project_images:
            deleted, _ = image.coordinates.all().delete()
            total_deleted += deleted
        return Response(
            {"detail": f"Deleted {total_deleted} coordinate entries from the project."},
            status=status.HTTP_200_OK,
        )
