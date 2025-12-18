from django.http import JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from . import helper


class OcrTrainingJobView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, job_id: str):
        job = helper.TRAINING_JOBS.get(job_id)
        if not job or job.user_id != request.user.id:
            return JsonResponse(
                {"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND
            )
        return JsonResponse(
            {"job": helper._serialize_job(job, include_logs=True)},
            status=status.HTTP_200_OK,
        )
