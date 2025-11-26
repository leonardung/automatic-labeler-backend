from django.urls import include, path
from rest_framework.routers import DefaultRouter
from api.views import ImageViewSet, MaskCategoryViewSet, ModelManagerViewSet, ProjectViewSet, VideoViewSet

router = DefaultRouter()
router.register(r"projects", ProjectViewSet, basename="project")
router.register(r"images", ImageViewSet, basename="images")
router.register(r"video", VideoViewSet, basename="video")
router.register(r"model", ModelManagerViewSet, basename="model")
router.register(r"categories", MaskCategoryViewSet, basename="categories")

urlpatterns = [
    path("", include(router.urls)),
]
