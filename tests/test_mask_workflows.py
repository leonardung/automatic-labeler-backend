import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.db import connections
from django.test import override_settings
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

# Provide a light cv2 stub for environments where OpenCV is unavailable.
if "cv2" not in sys.modules:
    import types

    cv2_stub = types.SimpleNamespace(IMREAD_COLOR=1, INTER_AREA=3)

    def _imdecode_stub(np_arr, flags):
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def _resize_stub(img, size, interpolation=None):
        return np.zeros((size[1], size[0], img.shape[2]), dtype=img.dtype)

    def _imencode_stub(ext, img):
        return True, np.ones((1,), dtype=np.uint8)

    cv2_stub.imdecode = _imdecode_stub
    cv2_stub.resize = _resize_stub
    cv2_stub.imencode = _imencode_stub
    sys.modules["cv2"] = cv2_stub

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import api.views as views
from api.models.image import ImageModel
from api.models.mask import MaskCategory, SegmentationMask
from api.models.project import Project


class MaskWorkflowTests(APITestCase):
    def setUp(self):
        # Each test gets its own SQLite database file that is removed at teardown.
        self.temp_media_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_media_root, True)
        temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(temp_db_fd)
        self.temp_db_path = temp_db_path

        base_db = settings.DATABASES["default"].copy()
        base_db["NAME"] = self.temp_db_path
        self.override_media = override_settings(MEDIA_ROOT=self.temp_media_root)
        self.override_db = override_settings(DATABASES={"default": base_db})
        self.override_media.enable()
        self.override_db.enable()
        connections.databases["default"] = base_db
        connections.close_all()
        call_command("migrate", verbosity=0, interactive=False)

        super().setUp()
        self.addCleanup(self._cleanup_db)

        self.user = User.objects.create_user(username="tester", password="password123")
        self.client = APIClient()
        self.client.force_authenticate(self.user)

        # Reset shared SAM3 caches so each test starts fresh.
        views.sam3_video_model = None
        views.sam3_video_states = {}
        views.sam3_text_obj_map = {}
        views.sam3_obj_category_map = {}
        views.propagation_progress = {}

    def _cleanup_db(self):
        connections.close_all()
        self.override_db.disable()
        self.override_media.disable()
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def _upload_images(
        self, project: Project, asset_subdir: str, filenames: List[str]
    ) -> List[int]:
        asset_base = (
            Path(settings.BASE_DIR) / "tests" / "asset" / asset_subdir / "images"
        )
        uploads = []
        for name in filenames:
            with open(asset_base / name, "rb") as fp:
                content_type = (
                    "image/png" if name.lower().endswith("png") else "image/jpeg"
                )
                uploads.append(
                    SimpleUploadedFile(
                        name=name, content=fp.read(), content_type=content_type
                    )
                )

        payload = {"project_id": project.id, "images": uploads}
        response = self.client.post("/api/images/", data=payload, format="multipart")
        self.assertEqual(
            response.status_code, status.HTTP_201_CREATED, response.content
        )
        return [img["id"] for img in response.data]

    @staticmethod
    def _mask_area(mask_path: str) -> int:
        with Image.open(mask_path) as img:
            arr = np.array(img)
        return int(np.count_nonzero(arr))

    def test_image_project_point_and_text_masks(self):
        project = Project.objects.create(
            user=self.user, name="Image Project", type="segmentation"
        )
        image_id = self._upload_images(project, "image_project", ["logs.png"])[0]

        coord = {"x": 180.2, "y": 247.4, "include": True}
        mask_resp = self.client.post(
            f"/api/images/{image_id}/generate_mask/",
            {"coordinates": [coord]},
            format="json",
        )
        self.assertEqual(mask_resp.status_code, status.HTTP_200_OK, mask_resp.content)

        image = ImageModel.objects.get(pk=image_id)
        default_mask = image.masks.first()
        self.assertIsNotNone(default_mask)
        area = self._mask_area(default_mask.mask.path)
        print(f"Image project 'single log' mask area: {area}")
        self.assertGreater(area, 0)

        text_resp = self.client.post(
            f"/api/images/{image_id}/generate_text_mask/",
            {"prompt": "wood log", "max_masks": 10, "threshold": 0.5},
            format="json",
        )
        self.assertEqual(text_resp.status_code, status.HTTP_200_OK, text_resp.content)
        self.assertEqual(len(text_resp.data.get("created_categories", [])), 10)
        log_masks = SegmentationMask.objects.filter(
            image_id=image_id, category__name__startswith="wood log_"
        )
        self.assertEqual(log_masks.count(), 10)

    def test_video_project_masks_propagation_and_text_prompts(self):
        project = Project.objects.create(
            user=self.user, name="Video Project", type="video_tracking_segmentation"
        )
        frame_names = ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg", "00004.jpg"]
        areas = {
            "ball": [140, 200],
            "player1": [42000, 47000],
            "player2": [12000, 14000],
        }
        frame_ids = self._upload_images(project, "video_project", frame_names)
        first_frame_id = frame_ids[0]

        coord = {"x": 735.6, "y": 434.9, "include": True}
        mask_resp = self.client.post(
            f"/api/images/{first_frame_id}/generate_mask/",
            {"coordinates": [coord]},
            format="json",
        )
        self.assertEqual(mask_resp.status_code, status.HTTP_200_OK, mask_resp.content)
        ball_mask = SegmentationMask.objects.get(image_id=first_frame_id)
        base_area = self._mask_area(ball_mask.mask.path)
        print(f"Video project 'ball' mask area (frame 0): {base_area}")
        self.assertGreater(base_area, 0)

        propagate_resp = self.client.post(
            "/api/images/propagate_mask/",
            {"project_id": project.id, "category_id": ball_mask.category_id},
            format="json",
        )
        self.assertEqual(
            propagate_resp.status_code, status.HTTP_200_OK, propagate_resp.content
        )
        ball_masks = SegmentationMask.objects.filter(
            category=ball_mask.category, image__project=project
        ).order_by("image__original_filename")
        self.assertEqual(ball_masks.count(), len(frame_ids))
        for mask in ball_masks:
            area = self._mask_area(mask.mask.path)
            print(
                f"Video project 'ball' mask area ({mask.image.original_filename}): {area}"
            )
            self.assertAlmostEqual(area, base_area, delta=50)
            self.assertGreaterEqual()
            self.assertLessEqual()

        text_resp = self.client.post(
            f"/api/images/{first_frame_id}/generate_text_mask/",
            {"prompt": "player", "max_masks": 10, "threshold": 0.5},
            format="json",
        )
        self.assertEqual(text_resp.status_code, status.HTTP_200_OK, text_resp.content)
        self.assertEqual(len(text_resp.data.get("created_categories", [])), 2)
        player_categories = MaskCategory.objects.filter(
            project=project, name__startswith="player_"
        )
        self.assertEqual(player_categories.count(), 2)

        propagate_players = self.client.post(
            "/api/images/propagate_mask/",
            {"project_id": project.id},
            format="json",
        )
        self.assertEqual(
            propagate_players.status_code, status.HTTP_200_OK, propagate_players.content
        )
        for category in player_categories:
            player_masks = SegmentationMask.objects.filter(
                category=category, image__project=project
            ).order_by("image__original_filename")
            self.assertEqual(player_masks.count(), len(frame_ids))
            first_area = self._mask_area(player_masks.first().mask.path)
            for mask in player_masks:
                area = self._mask_area(mask.mask.path)
                self.assertAlmostEqual(area, first_area, delta=3000)
