"""Depth Anything model worker for depth estimation."""

import logging
from typing import Optional

import numpy as np
from PIL import Image

from modelstag.core.types import ModelType
from modelstag.config.models import ModelConfig
from modelstag.workers.base import BaseModelWorker
from modelstag.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)

DEPTH_MODELS = {
    "small": "LiheYoung/depth-anything-small-hf",
    "base": "LiheYoung/depth-anything-base-hf",
    "large": "LiheYoung/depth-anything-large-hf",
}


@WorkerRegistry.register(ModelType.DEPTH)
class DepthWorker(BaseModelWorker):
    """Worker for Depth Anything depth estimation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._pipe = None
        self._model_size = getattr(config, 'depth_model', 'small') or 'small'

    def load_model(self) -> None:
        """Load the Depth Anything model."""
        if self._loaded:
            return

        import torch
        from transformers import pipeline

        logger.info(f"Loading Depth Anything model: {self._model_size}")

        model_id = DEPTH_MODELS.get(self._model_size, DEPTH_MODELS["small"])

        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._pipe = pipeline(
            "depth-estimation",
            model=model_id,
            device=device,
        )

        self._loaded = True
        logger.info(f"Loaded Depth Anything model: {self._model_size} on {device}")

    def unload_model(self) -> None:
        """Unload the model."""
        if not self._loaded:
            return

        logger.info(f"Unloading Depth Anything model: {self._model_size}")
        self._pipe = None
        self._loaded = False

    def predict(self, image: np.ndarray, **options) -> np.ndarray:
        """Run depth estimation.

        Returns a depth map as grayscale image (darker = closer, lighter = farther).
        """
        if not self._loaded or self._pipe is None:
            raise RuntimeError("Model not loaded")

        pil_image = Image.fromarray(image)

        # Run depth estimation
        result = self._pipe(pil_image)
        depth_map = result["depth"]

        # Convert to numpy and normalize to 0-255
        depth_array = np.array(depth_map)
        depth_normalized = ((depth_array - depth_array.min()) /
                          (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)

        return depth_normalized

    def predict_raw(self, image: np.ndarray, **options) -> dict:
        """Run depth estimation and return raw depth values.

        Returns dict with 'depth_map' (normalized 0-255) and 'depth_raw' (original values).
        """
        if not self._loaded or self._pipe is None:
            raise RuntimeError("Model not loaded")

        pil_image = Image.fromarray(image)
        result = self._pipe(pil_image)
        depth_map = result["depth"]
        depth_array = np.array(depth_map)

        depth_normalized = ((depth_array - depth_array.min()) /
                          (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)

        return {
            'depth_map': depth_normalized,
            'depth_min': float(depth_array.min()),
            'depth_max': float(depth_array.max()),
            'width': depth_array.shape[1],
            'height': depth_array.shape[0],
        }
