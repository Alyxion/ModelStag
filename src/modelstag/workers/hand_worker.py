"""Hand tracking worker supporting MediaPipe Hands and HaMeR."""

import logging
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

# MediaPipe hand model URL
MEDIAPIPE_HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# MediaPipe hand landmark names (21 landmarks per hand)
HAND_LANDMARKS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# Hand skeleton connections
HAND_SKELETON = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]


def get_mediapipe_hand_model_path() -> Path:
    """Get path to MediaPipe hand model, downloading if necessary."""
    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / "hand_landmarker.task"

    if not model_path.exists():
        logger.info("Downloading MediaPipe Hand model...")
        urllib.request.urlretrieve(MEDIAPIPE_HAND_MODEL_URL, model_path)
        logger.info(f"Downloaded to {model_path}")

    return model_path


@WorkerRegistry.register(ModelType.HAND)
class HandWorker(BaseModelWorker):
    """Worker for hand tracking using MediaPipe Hands or HaMeR."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._model_type = getattr(config, 'hand_model', 'mediapipe') or 'mediapipe'
        self._device = None

    def load_model(self) -> None:
        """Load the hand tracking model."""
        if self._loaded:
            return

        logger.info(f"Loading hand model: {self._model_type}")

        if self._model_type == "mediapipe":
            self._load_mediapipe()
        elif self._model_type == "hamer":
            self._load_hamer()
        else:
            raise ValueError(f"Unknown hand model type: {self._model_type}")

        self._loaded = True
        logger.info(f"Loaded hand model: {self._model_type}")

    def _load_mediapipe(self) -> None:
        """Load MediaPipe Hands model using the new Tasks API."""
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = get_mediapipe_hand_model_path()

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=4,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._model = vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def _load_hamer(self) -> None:
        """Load HaMeR model for 3D hand mesh reconstruction."""
        try:
            import torch

            # Check for GPU
            self._device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )

            # Try to import hamer
            from hamer.models import HAMER
            from hamer.configs import CACHE_DIR_HAMER

            # Load pretrained model
            self._model = HAMER.from_pretrained(
                "geopavlakos/hamer",
                cache_dir=CACHE_DIR_HAMER,
            ).to(self._device)
            self._model.eval()

        except ImportError as e:
            raise ImportError(
                f"HaMeR requires additional dependencies. Install with:\n"
                f"pip install git+https://github.com/geopavlakos/hamer.git\n"
                f"Original error: {e}"
            )

    def unload_model(self) -> None:
        """Unload the model."""
        if not self._loaded:
            return

        logger.info(f"Unloading hand model: {self._model_type}")

        if self._model_type == "mediapipe" and self._model:
            self._model.close()

        self._model = None
        self._loaded = False

    def predict(self, image: np.ndarray, **options) -> np.ndarray:
        """Run hand tracking and return a visualization mask."""
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        result = self.predict_landmarks(image, **options)

        # Create visualization mask
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw landmarks on mask
        for hand_idx, hand in enumerate(result.get('hands', [])):
            landmarks = hand.get('landmarks', [])
            hand_value = (hand_idx + 1) * 50  # Different intensity per hand

            for i, lm in enumerate(landmarks):
                if lm.get('confidence', 1.0) > 0.3:
                    x, y = int(lm['x']), int(lm['y'])
                    if 0 <= x < w and 0 <= y < h:
                        cv2_circle(mask, x, y, 3, hand_value + i)

        return mask

    def predict_landmarks(self, image: np.ndarray, **options) -> dict:
        """Run hand tracking and return landmarks.

        Returns:
            Dict with 'hands' list, each containing 'landmarks' with x, y, z, confidence,
            and 'handedness' (left/right).
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        if self._model_type == "mediapipe":
            return self._predict_mediapipe(image, **options)
        elif self._model_type == "hamer":
            return self._predict_hamer(image, **options)
        else:
            raise RuntimeError(f"Unknown model type: {self._model_type}")

    def _predict_mediapipe(self, image: np.ndarray, **options) -> dict:
        """Run MediaPipe hands detection."""
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

        hands = []
        h, w = image.shape[:2]

        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                landmarks = []

                for i, landmark in enumerate(hand_landmarks):
                    landmarks.append({
                        'x': landmark.x * w,
                        'y': landmark.y * h,
                        'z': landmark.z,
                        'confidence': 1.0,  # MediaPipe doesn't provide per-landmark confidence
                        'name': HAND_LANDMARKS[i] if i < len(HAND_LANDMARKS) else f"landmark_{i}",
                    })

                # Get handedness (left/right)
                handedness = "unknown"
                if result.handedness and hand_idx < len(result.handedness):
                    handedness = result.handedness[hand_idx][0].category_name.lower()
                    hand_score = result.handedness[hand_idx][0].score
                else:
                    hand_score = 1.0

                hands.append({
                    'landmarks': landmarks,
                    'handedness': handedness,
                    'score': hand_score,
                })

        return {
            'hands': hands,
            'count': len(hands),
            'model': "mediapipe-hands",
            'landmark_names': HAND_LANDMARKS,
            'skeleton': HAND_SKELETON,
        }

    def _predict_hamer(self, image: np.ndarray, **options) -> dict:
        """Run HaMeR 3D hand mesh reconstruction."""
        import torch
        from PIL import Image as PILImage

        # Convert to PIL
        pil_image = PILImage.fromarray(image)

        hands = []

        # Run inference
        with torch.no_grad():
            try:
                # HaMeR inference - simplified
                # Full implementation requires the complete HaMeR pipeline
                output = self._model(pil_image)

                if output is not None:
                    for i, pred in enumerate(output):
                        if hasattr(pred, 'joints_2d'):
                            landmarks = []
                            for j, joint in enumerate(pred.joints_2d):
                                landmarks.append({
                                    'x': float(joint[0]),
                                    'y': float(joint[1]),
                                    'z': float(pred.joints_3d[j, 2]) if hasattr(pred, 'joints_3d') else 0.0,
                                    'confidence': 1.0,
                                    'name': HAND_LANDMARKS[j] if j < len(HAND_LANDMARKS) else f"joint_{j}",
                                })

                            hands.append({
                                'landmarks': landmarks,
                                'handedness': pred.handedness if hasattr(pred, 'handedness') else "unknown",
                                'score': float(pred.score) if hasattr(pred, 'score') else 1.0,
                                'mano_params': {
                                    'global_orient': pred.global_orient.tolist() if hasattr(pred, 'global_orient') else None,
                                    'hand_pose': pred.hand_pose.tolist() if hasattr(pred, 'hand_pose') else None,
                                    'betas': pred.betas.tolist() if hasattr(pred, 'betas') else None,
                                },
                                'vertices': pred.vertices.tolist() if hasattr(pred, 'vertices') else None,
                            })
            except Exception as e:
                logger.warning(f"HaMeR inference failed: {e}")

        return {
            'hands': hands,
            'count': len(hands),
            'model': "hamer",
            'landmark_names': HAND_LANDMARKS,
            'skeleton': HAND_SKELETON,
            'has_3d_mesh': True,
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
