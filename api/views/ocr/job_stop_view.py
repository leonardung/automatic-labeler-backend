import subprocess

from django.http import JsonResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from . import helper


class OcrTrainingJobStopView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def post(self, request, job_id: str):
        job = helper.TRAINING_JOBS.get(job_id)
        if not job or job.user_id != request.user.id:
            return JsonResponse(
                {"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND
            )
        _stop_job(job)
        return JsonResponse(
            {"job": helper._serialize_job(job)}, status=status.HTTP_200_OK
        )


def _stop_job(job: helper.TrainingJob):
    if job.status in ("completed", "failed", "stopped"):
        return
    job.stop_requested = True
    job.append_log("Stop requested by user.\n")
    with helper.TRAINING_LOCK:
        if job.status == "waiting":
            try:
                helper.TRAINING_QUEUE.remove(job.id)
            except ValueError:
                pass
            helper._finish_job(job, "stopped", error="Stopped by user.")
            return
    if job.status == "running" and job.current_process:
        proc = job.current_process
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
