"""Base model worker abstract class."""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from modelstag.config.models import ModelConfig
from modelstag.outputs.base import MaskOutput, InferenceResult
from modelstag.outputs.converters import mask_to_polygons, mask_to_bbox

logger = logging.getLogger(__name__)


class BaseModelWorker(ABC):
    """Abstract base class for model workers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        self._loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray, **options) -> np.ndarray:
        """Run inference on image.

        Args:
            image: Input image as numpy array (RGB, uint8)
            **options: Model-specific options

        Returns:
            Mask as numpy array (grayscale, uint8)
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def process_request(
        self,
        image: np.ndarray,
        output_formats: List[str],
        **options,
    ) -> InferenceResult:
        """Process an inference request.

        Args:
            image: Input image
            output_formats: List of desired output formats
            **options: Model-specific options

        Returns:
            InferenceResult with requested formats
        """
        start_time = time.perf_counter()

        try:
            # Run inference
            mask_array = self.predict(image, **options)

            # Create result
            result = InferenceResult(model_name=self.name)

            # Convert to requested formats
            mask_output = MaskOutput.from_array(mask_array)

            if "mask" in output_formats:
                result.mask = mask_output

            # Use grayscale version for polygon/bbox extraction
            gray_mask = mask_output.to_grayscale()

            if "polygon" in output_formats:
                result.polygons = mask_to_polygons(gray_mask)

            if "bbox" in output_formats:
                result.bbox = mask_to_bbox(gray_mask)

            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        except Exception as e:
            logger.exception(f"Inference error in {self.name}")
            return InferenceResult(
                model_name=self.name,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )
