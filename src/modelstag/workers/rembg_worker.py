"""Rembg model worker implementation."""

import logging
from typing import Optional
import numpy as np
from PIL import Image

# Import rembg at module load to avoid import lock issues in threads
from rembg import new_session as rembg_new_session
from rembg import remove as rembg_remove

from modelstag.core.types import ModelType
from modelstag.config.models import ModelConfig
from modelstag.workers.base import BaseModelWorker
from modelstag.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)


@WorkerRegistry.register(ModelType.REMBG)
class RembgWorker(BaseModelWorker):
    """Worker for rembg background removal models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._session = None
        self._model_name = config.rembg_model or "u2net"

    def load_model(self) -> None:
        """Load the rembg session."""
        if self._loaded:
            return

        logger.info(f"Loading rembg model: {self._model_name}")
        self._session = rembg_new_session(self._model_name)
        self._loaded = True
        logger.info(f"Loaded rembg model: {self._model_name}")

    def unload_model(self) -> None:
        """Unload the rembg session."""
        if not self._loaded:
            return

        logger.info(f"Unloading rembg model: {self._model_name}")
        self._session = None
        self._loaded = False
        logger.info(f"Unloaded rembg model: {self._model_name}")

    def predict(self, image: np.ndarray, **options) -> np.ndarray:
        """Run background removal inference."""
        if not self._loaded or self._session is None:
            raise RuntimeError("Model not loaded")

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Limit input size for efficiency (models process at 320-1024px internally)
        max_size = 1024
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized input from {image.shape[:2]} to {new_size}")

        # Run inference
        result = rembg_remove(
            pil_image,
            session=self._session,
            only_mask=True,
            **options,
        )

        # Convert back to numpy array
        mask = np.array(result)

        # Ensure grayscale
        if mask.ndim == 3:
            if mask.shape[2] == 4:
                mask = mask[:, :, 3]  # Use alpha channel
            else:
                mask = mask[:, :, 0]

        return mask
