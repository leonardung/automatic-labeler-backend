from django.shortcuts import get_object_or_404
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

from api.models.mask import MaskCategory
from api.models.project import Project
from api.serializers import MaskCategorySerializer


class MaskCategoryViewSet(viewsets.ModelViewSet):
    queryset = MaskCategory.objects.all()
    serializer_class = MaskCategorySerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get_queryset(self):
        return MaskCategory.objects.filter(project__user=self.request.user)

    def perform_create(self, serializer):
        project_id = self.request.data.get("project_id")
        project = get_object_or_404(Project, id=project_id, user=self.request.user)
        serializer.save(project=project)
