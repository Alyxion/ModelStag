"""Segment Anything Model (SAM) worker implementation."""

import logging
import os
from pathlib import Path
from typing import List, Optional
import urllib.request

import numpy as np
from PIL import Image

from modelstag.core.types import ModelType
from modelstag.config.models import ModelConfig
from modelstag.workers.base import BaseModelWorker
from modelstag.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)

# Model checkpoints
SAM_MODELS = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "size": "375MB",
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "size": "1.2GB",
    },
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "size": "2.4GB",
    },
}


def get_checkpoint_path(model_type: str = "vit_b") -> Path:
    """Get path to SAM checkpoint, downloading if necessary."""
    cache_dir = Path.home() / ".cache" / "sam"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_info = SAM_MODELS.get(model_type, SAM_MODELS["vit_b"])
    filename = model_info["url"].split("/")[-1]
    checkpoint_path = cache_dir / filename

    if not checkpoint_path.exists():
        logger.info(f"Downloading SAM {model_type} checkpoint ({model_info['size']})...")
        urllib.request.urlretrieve(model_info["url"], checkpoint_path)
        logger.info(f"Downloaded to {checkpoint_path}")

    return checkpoint_path


@WorkerRegistry.register(ModelType.SAM)
class SamWorker(BaseModelWorker):
    """Worker for Segment Anything Model with automatic mask generation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._sam = None
        self._mask_generator = None
        self._model_type = getattr(config, 'sam_model', 'vit_b') or 'vit_b'

    def load_model(self) -> None:
        """Load the SAM model."""
        if self._loaded:
            return

        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        logger.info(f"Loading SAM model: {self._model_type}")

        checkpoint_path = get_checkpoint_path(self._model_type)

        # Load model
        self._sam = sam_model_registry[self._model_type](checkpoint=str(checkpoint_path))

        # Use CPU (add CUDA support later if needed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sam.to(device=device)

        # Create automatic mask generator with reasonable defaults
        self._mask_generator = SamAutomaticMaskGenerator(
            model=self._sam,
            points_per_side=32,  # Grid density (32x32 = 1024 points)
            pred_iou_thresh=0.88,  # Confidence threshold
            stability_score_thresh=0.95,  # Mask stability threshold
            min_mask_region_area=100,  # Filter tiny masks
        )

        self._loaded = True
        logger.info(f"Loaded SAM model: {self._model_type} on {device}")

    def unload_model(self) -> None:
        """Unload the SAM model."""
        if not self._loaded:
            return

        logger.info(f"Unloading SAM model: {self._model_type}")
        self._mask_generator = None
        self._sam = None
        self._loaded = False
        logger.info(f"Unloaded SAM model: {self._model_type}")

    def predict(self, image: np.ndarray, **options) -> np.ndarray:
        """Run automatic mask generation.

        Returns a labeled mask where each object has a unique ID.
        """
        if not self._loaded or self._mask_generator is None:
            raise RuntimeError("Model not loaded")

        # Limit input size for performance
        pil_image = Image.fromarray(image)
        max_size = 1024
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            image = np.array(pil_image)

        # Generate all masks
        masks = self._mask_generator.generate(image)

        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Create labeled output (each segment gets unique ID)
        h, w = image.shape[:2]
        labeled = np.zeros((h, w), dtype=np.uint8)

        # Assign labels (1-255, 0 is background)
        for i, mask_data in enumerate(masks[:255]):  # Max 255 objects
            mask = mask_data['segmentation']
            labeled[mask] = i + 1

        return labeled

    def predict_all_masks(
        self,
        image: np.ndarray,
        max_masks: int = 100,
        max_total_bytes: int = 20 * 1024 * 1024,  # 20 MB
        **options,
    ) -> dict:
        """Run automatic mask generation and return all masks with metadata.

        Args:
            image: Input image as numpy array.
            max_masks: Maximum number of masks to return (default: 100).
            max_total_bytes: Maximum total covered area in pixels (default: 20M pixels).

        Returns:
            Dict with 'masks' list and 'stats' dict containing generation statistics.
            Each mask dict has 'mask', 'area', 'bbox', 'score'.
        """
        if not self._loaded or self._mask_generator is None:
            raise RuntimeError("Model not loaded")

        # Limit input size
        pil_image = Image.fromarray(image)
        max_size = 1024
        scale = 1.0
        if max(pil_image.size) > max_size:
            scale = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            image = np.array(pil_image)

        # Generate all masks
        masks = self._mask_generator.generate(image)
        total_generated = len(masks)

        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        # Return structured data with limits
        results = []
        total_bytes = 0
        masks_limited_by_count = False
        masks_limited_by_memory = False

        for mask_data in masks:
            # Check mask count limit
            if len(results) >= max_masks:
                masks_limited_by_count = True
                break

            # Calculate mask memory based on actual covered area (not full image size)
            mask_bytes = mask_data['area']  # actual pixels in mask
            if total_bytes + mask_bytes > max_total_bytes:
                masks_limited_by_memory = True
                break

            total_bytes += mask_bytes
            results.append({
                'mask': (mask_data['segmentation'] * 255).astype(np.uint8),
                'area': mask_data['area'],
                'bbox': mask_data['bbox'],  # [x, y, w, h]
                'score': mask_data['predicted_iou'],
            })

        return {
            'masks': results,
            'stats': {
                'total_generated': total_generated,
                'returned': len(results),
                'limited_by_count': masks_limited_by_count,
                'limited_by_memory': masks_limited_by_memory,
                'total_bytes': total_bytes,
                'max_masks': max_masks,
                'max_total_bytes': max_total_bytes,
            },
        }
