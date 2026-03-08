"""Pose estimation worker supporting MediaPipe Pose, RTMO, and RTMW."""

import logging
import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from PIL import Image

from modelstag.core.types import ModelType
from modelstag.config.models import ModelConfig
from modelstag.workers.base import BaseModelWorker
from modelstag.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)

# MediaPipe model URLs
MEDIAPIPE_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MEDIAPIPE_POSE_MODEL_LITE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

# RTMO/RTMW models from MMPose
RTMO_MODELS = {
    "rtmo_s": "rtmo-s_8xb32-600e_body7-640x640",
    "rtmo_m": "rtmo-m_16xb16-600e_body7-640x640",
    "rtmo_l": "rtmo-l_16xb16-600e_body7-640x640",
}

RTMW_MODELS = {
    "rtmw_m": "rtmw-m_8xb64-270e_cocktail14-256x192",
    "rtmw_l": "rtmw-l_8xb64-270e_cocktail14-256x192",
}

# MediaPipe pose landmark names (33 landmarks)
MEDIAPIPE_POSE_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# COCO keypoint names (17 keypoints)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def get_mediapipe_model_path() -> Path:
    """Get path to MediaPipe pose model, downloading if necessary."""
    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / "pose_landmarker_heavy.task"

    if not model_path.exists():
        logger.info("Downloading MediaPipe Pose model...")
        urllib.request.urlretrieve(MEDIAPIPE_POSE_MODEL_URL, model_path)
        logger.info(f"Downloaded to {model_path}")

    return model_path


@WorkerRegistry.register(ModelType.POSE)
class PoseWorker(BaseModelWorker):
    """Worker for pose estimation using MediaPipe, RTMO, or RTMW."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._model_type = getattr(config, 'pose_model', 'mediapipe') or 'mediapipe'
        self._device = None

    def load_model(self) -> None:
        """Load the pose estimation model."""
        if self._loaded:
            return

        logger.info(f"Loading pose model: {self._model_type}")

        if self._model_type == "mediapipe":
            self._load_mediapipe()
        elif self._model_type.startswith("rtmo"):
            self._load_rtmo()
        elif self._model_type.startswith("rtmw"):
            self._load_rtmw()
        else:
            raise ValueError(f"Unknown pose model type: {self._model_type}")

        self._loaded = True
        logger.info(f"Loaded pose model: {self._model_type}")

    def _load_mediapipe(self) -> None:
        """Load MediaPipe Pose model using the new Tasks API."""
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = get_mediapipe_model_path()

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            num_poses=5,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._model = vision.PoseLandmarker.create_from_options(options)
        self._mp = mp

    def _load_rtmo(self) -> None:
        """Load RTMO model via MMPose."""
        try:
            from mmpose.apis import MMPoseInferencer

            model_name = RTMO_MODELS.get(self._model_type, RTMO_MODELS["rtmo_s"])
            self._model = MMPoseInferencer(pose2d=model_name)
            self._inferencer_type = "mmpose"
        except ImportError:
            raise ImportError(
                "RTMO requires mmpose. Install with: pip install mmpose mmdet mmcv mmengine"
            )

    def _load_rtmw(self) -> None:
        """Load RTMW wholebody model via MMPose."""
        try:
            from mmpose.apis import MMPoseInferencer

            model_name = RTMW_MODELS.get(self._model_type, RTMW_MODELS["rtmw_m"])
            self._model = MMPoseInferencer(pose2d=model_name)
            self._inferencer_type = "mmpose"
        except ImportError:
            raise ImportError(
                "RTMW requires mmpose. Install with: pip install mmpose mmdet mmcv mmengine"
            )

    def unload_model(self) -> None:
        """Unload the model."""
        if not self._loaded:
            return

        logger.info(f"Unloading pose model: {self._model_type}")

        if self._model_type == "mediapipe" and self._model:
            self._model.close()

        self._model = None
        self._loaded = False

    def predict(self, image: np.ndarray, **options) -> np.ndarray:
        """Run pose estimation and return a visualization mask."""
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        result = self.predict_keypoints(image, **options)

        # Create visualization mask
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw keypoints on mask
        for person_idx, person in enumerate(result.get('poses', [])):
            keypoints = person.get('keypoints', [])
            for i, kp in enumerate(keypoints):
                if kp.get('confidence', 0) > 0.3:
                    x, y = int(kp['x']), int(kp['y'])
                    if 0 <= x < w and 0 <= y < h:
                        cv2_circle(mask, x, y, 3, (person_idx + 1) * 10 + i % 10)

        return mask

    def predict_keypoints(self, image: np.ndarray, **options) -> dict:
        """Run pose estimation and return keypoints.

        Returns:
            Dict with 'poses' list, each containing 'keypoints' with x, y, z, confidence, name.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        if self._model_type == "mediapipe":
            return self._predict_mediapipe(image, **options)
        elif hasattr(self, '_inferencer_type') and self._inferencer_type == "mmpose":
            return self._predict_mmpose(image, **options)
        else:
            raise RuntimeError(f"Unknown model type: {self._model_type}")

    def _predict_mediapipe(self, image: np.ndarray, **options) -> dict:
        """Run MediaPipe pose estimation."""
        import cv2

        # MediaPipe expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Create MediaPipe Image
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)

        # Run detection
        result = self._model.detect(mp_image)

        poses = []
        h, w = image.shape[:2]

        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:
                keypoints = []

                for i, landmark in enumerate(pose_landmarks):
                    keypoints.append({
                        'x': landmark.x * w,
                        'y': landmark.y * h,
                        'z': landmark.z,
                        'confidence': landmark.visibility if hasattr(landmark, 'visibility') else 1.0,
                        'name': MEDIAPIPE_POSE_LANDMARKS[i] if i < len(MEDIAPIPE_POSE_LANDMARKS) else f"landmark_{i}",
                    })

                poses.append({
                    'keypoints': keypoints,
                    'score': np.mean([kp['confidence'] for kp in keypoints]),
                })

        return {
            'poses': poses,
            'count': len(poses),
            'model': "mediapipe-pose",
            'keypoint_names': MEDIAPIPE_POSE_LANDMARKS,
        }

    def _predict_mmpose(self, image: np.ndarray, **options) -> dict:
        """Run MMPose inference (RTMO/RTMW)."""
        result_generator = self._model(image, return_datasamples=True)
        result = next(result_generator)

        poses = []
        if 'predictions' in result:
            for pred in result['predictions']:
                if hasattr(pred, 'pred_instances'):
                    instances = pred.pred_instances
                    for i in range(len(instances.keypoints)):
                        kps = instances.keypoints[i]
                        scores = instances.keypoint_scores[i] if hasattr(instances, 'keypoint_scores') else [1.0] * len(kps)

                        keypoints = []
                        for j, (kp, score) in enumerate(zip(kps, scores)):
                            keypoints.append({
                                'x': float(kp[0]),
                                'y': float(kp[1]),
                                'confidence': float(score),
                                'name': COCO_KEYPOINTS[j] if j < len(COCO_KEYPOINTS) else f"keypoint_{j}",
                            })

                        poses.append({
                            'keypoints': keypoints,
                            'score': float(instances.scores[i]) if hasattr(instances, 'scores') else 1.0,
                            'bbox': instances.bboxes[i].tolist() if hasattr(instances, 'bboxes') else None,
                        })

        return {
            'poses': poses,
            'count': len(poses),
            'model': self._model_type,
            'keypoint_names': COCO_KEYPOINTS,
        }


def cv2_circle(mask: np.ndarray, x: int, y: int, radius: int, value: int) -> None:
    """Draw a filled circle on mask without importing cv2."""
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    mask[ny, nx] = value
