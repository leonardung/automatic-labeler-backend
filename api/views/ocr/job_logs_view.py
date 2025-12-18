from django.http import FileResponse, JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from . import helper


class OcrTrainingJobLogsView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, job_id: str):
        job = helper.TRAINING_JOBS.get(job_id)
        if not job or job.user_id != request.user.id:
            return JsonResponse(
                {"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND
            )
        if not job.log_path or not job.log_path.exists():
            return JsonResponse(
                {"error": "Log not found for this job."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return FileResponse(
            job.log_path.open("rb"), as_attachment=True, filename=f"{job.id}.log"
        )
