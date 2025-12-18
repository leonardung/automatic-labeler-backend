from django.http import JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from . import helper


class OcrTrainingJobListView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request):
        jobs = [
            job for job in helper.TRAINING_JOBS.values() if job.user_id == request.user.id
        ]
        jobs_sorted = sorted(jobs, key=lambda j: j.created_at, reverse=True)
        return JsonResponse(
            {"jobs": [helper._serialize_job(job) for job in jobs_sorted]},
            status=status.HTTP_200_OK,
        )
