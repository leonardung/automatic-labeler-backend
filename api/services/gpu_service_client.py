"""
GPU Service Client

Client for communicating with the GPU microservice for OCR operations.
"""

import logging
import os
from typing import Dict, List, Optional, Any

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class GPUServiceError(Exception):
    """Exception raised when GPU service returns an error"""

    pass


class GPUServiceClient:
    """Client for GPU OCR microservice"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Initialize GPU service client.

        Args:
            base_url: Base URL of GPU service (default: from settings)
            token: Authentication token (default: from settings)
            timeout: Request timeout in seconds (default: 300)
        """
        self.base_url = (
            base_url
            or getattr(settings, "GPU_SERVICE_URL", None)
            or os.getenv("GPU_SERVICE_URL", "http://localhost:8001")
        )
        self.token = (
            token
            or getattr(settings, "GPU_SERVICE_TOKEN", None)
            or os.getenv("GPU_SERVICE_TOKEN")
        )
        self.timeout = timeout

        if not self.token:
            logger.warning("GPU_SERVICE_TOKEN not configured")

        self.base_url = self.base_url.rstrip("/")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Make a request to the GPU service.

        Args:
            method: HTTP method
            endpoint: API endpoint
            json_data: JSON data to send
            params: URL parameters

        Returns:
            Response data as dict

        Raises:
            GPUServiceError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        try:
            response = requests.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("GPU service request timeout: %s %s", method, endpoint)
            raise GPUServiceError("GPU service request timeout")
        except requests.exceptions.ConnectionError as exc:
            logger.error("GPU service connection error: %s", exc)
            raise GPUServiceError("Unable to connect to GPU service")
        except requests.exceptions.HTTPError as exc:
            logger.error("GPU service HTTP error: %s", exc)
            try:
                error_data = exc.response.json()
                error_msg = error_data.get("detail", str(exc))
            except Exception:
                error_msg = str(exc)
            raise GPUServiceError(f"GPU service error: {error_msg}")
        except Exception as exc:
            logger.exception("Unexpected GPU service error: %s", exc)
            raise GPUServiceError(f"Unexpected error: {exc}")

    def health_check(self) -> bool:
        """
        Check if GPU service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self._request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception as exc:
            logger.error("GPU service health check failed: %s", exc)
            return False

    def detect_regions(
        self,
        image_path: str,
        model_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Detect text regions in an image.

        Args:
            image_path: Absolute path to image file
            model_name: Optional model name to use
            config: Optional detection configuration

        Returns:
            List of detected shapes

        Raises:
            GPUServiceError: If detection fails
        """
        payload = {"image_path": image_path}
        if model_name:
            payload["model_name"] = model_name
        if config:
            payload["config"] = config

        response = self._request("POST", "/api/v1/detect", json_data=payload)
        return response.get("shapes", [])

    def recognize_text(
        self,
        image_path: str,
        shapes: List[Dict],
        model_name: Optional[str] = None,
    ) -> Dict:
        """
        Recognize text in detected regions.

        Args:
            image_path: Absolute path to image file
            shapes: List of shapes to recognize text in
            model_name: Optional model name to use

        Returns:
            Dict with 'shapes' and optional 'categories'

        Raises:
            GPUServiceError: If recognition fails
        """
        payload = {
            "image_path": image_path,
            "shapes": shapes,
        }
        if model_name:
            payload["model_name"] = model_name

        return self._request("POST", "/api/v1/recognize", json_data=payload)

    def classify_kie(
        self,
        image_path: str,
        shapes: List[Dict],
        project_id: int,
        model_name: Optional[str] = None,
        checkpoint_type: str = "best",
        use_gpu: bool = True,
        max_len_per_part: int = 512,
        min_overlap: int = 64,
    ) -> Dict:
        """
        Classify KIE categories for detected text regions.

        Args:
            image_path: Absolute path to image file
            shapes: List of shapes to classify
            project_id: Project ID for model lookup
            model_name: Optional model name to use
            checkpoint_type: 'best' or 'latest'
            use_gpu: Whether to use GPU
            max_len_per_part: Max sequence length per chunk
            min_overlap: Minimum token overlap between chunks

        Returns:
            Dict with 'shapes' and 'categories'

        Raises:
            GPUServiceError: If classification fails
        """
        payload = {
            "image_path": image_path,
            "shapes": shapes,
            "project_id": project_id,
            "checkpoint_type": checkpoint_type,
            "use_gpu": use_gpu,
            "max_len_per_part": max_len_per_part,
            "min_overlap": min_overlap,
        }
        if model_name:
            payload["model_name"] = model_name

        return self._request("POST", "/api/v1/classify-kie", json_data=payload)

    def configure_models(
        self,
        detect_model: Optional[str] = None,
        recognize_model: Optional[str] = None,
        classify_model: Optional[str] = None,
        model_config: Optional[Dict] = None,
    ) -> Dict:
        """
        Configure active OCR models.

        Args:
            detect_model: Detection model name
            recognize_model: Recognition model name
            classify_model: Classification model name
            model_config: Model configuration dict

        Returns:
            Dict with active models and what changed

        Raises:
            GPUServiceError: If configuration fails
        """
        payload = {}
        if detect_model:
            payload["detect_model"] = detect_model
        if recognize_model:
            payload["recognize_model"] = recognize_model
        if classify_model:
            payload["classify_model"] = classify_model
        if model_config:
            payload["model_config"] = model_config

        return self._request("POST", "/api/v1/configure", json_data=payload)

    def get_active_models(self) -> Dict:
        """
        Get currently active models.

        Returns:
            Dict with active model names

        Raises:
            GPUServiceError: If request fails
        """
        return self._request("GET", "/api/v1/active")


# Global client instance
_client: Optional[GPUServiceClient] = None


def get_gpu_client() -> GPUServiceClient:
    """
    Get or create the global GPU service client instance.

    Returns:
        GPUServiceClient instance
    """
    global _client
    if _client is None:
        _client = GPUServiceClient()
    return _client
