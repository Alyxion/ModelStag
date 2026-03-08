"""Florence-2 image captioning and description worker."""

import logging
from typing import Optional

import numpy as np
from PIL import Image

from modelstag.core.types import ModelType
from modelstag.config.models import ModelConfig
from modelstag.workers.base import BaseModelWorker
from modelstag.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)

FLORENCE_MODELS = {
    "base": "microsoft/Florence-2-base",
    "large": "microsoft/Florence-2-large",
}

TASK_PROMPTS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "ocr": "<OCR>",
    "object_detection": "<OD>",
}


@WorkerRegistry.register(ModelType.CAPTION)
class CaptionWorker(BaseModelWorker):
    """Worker for Florence-2 image captioning and understanding."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._model_size = getattr(config, 'caption_model', 'base') or 'base'
        self._device = None

    def load_model(self) -> None:
        """Load the Florence-2 model."""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(f"Loading Florence-2 model: {self._model_size}")

        model_id = FLORENCE_MODELS.get(self._model_size, FLORENCE_MODELS["base"])

        self._device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Use float16 for CUDA, float32 for CPU/MPS (MPS has issues with float16 for Florence-2)
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        # Load processor first
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Load model with error handling for Florence-2 compatibility issues
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self._device)
        except AttributeError as e:
            if 'forced_bos_token_id' in str(e):
                # Workaround for Florence-2 compatibility issue with newer transformers
                logger.warning("Applying Florence-2 compatibility fix...")
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                # Set missing attribute
                if not hasattr(config.text_config, 'forced_bos_token_id'):
                    config.text_config.forced_bos_token_id = None
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(self._device)
            else:
                raise

        self._loaded = True
        logger.info(f"Loaded Florence-2 model: {self._model_size} on {self._device}")

    def unload_model(self) -> None:
        """Unload the model."""
        if not self._loaded:
            return

        logger.info(f"Unloading Florence-2 model: {self._model_size}")
        self._model = None
        self._processor = None
        self._loaded = False

    def predict(self, image: np.ndarray, **options) -> np.ndarray:
        """Run captioning - returns empty mask (captioning doesn't produce masks)."""
        # Caption models don't produce masks, return empty
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    def _run_inference(self, image: Image.Image, task: str) -> str:
        """Run Florence-2 inference for a given task."""
        import torch

        prompt = TASK_PROMPTS.get(task, task)

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self._processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height),
        )

        return parsed.get(prompt, generated_text)

    def caption(self, image: np.ndarray, detail_level: str = "detailed") -> dict:
        """Generate a caption for the image.

        Args:
            image: Input image as numpy array
            detail_level: One of "brief", "detailed", "verbose"

        Returns:
            Dict with 'caption' string and metadata.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        pil_image = Image.fromarray(image)

        task_map = {
            "brief": "caption",
            "detailed": "detailed_caption",
            "verbose": "more_detailed_caption",
        }
        task = task_map.get(detail_level, "detailed_caption")

        caption = self._run_inference(pil_image, task)

        return {
            'caption': caption,
            'detail_level': detail_level,
            'model': f"florence-2-{self._model_size}",
        }

    def ocr(self, image: np.ndarray) -> dict:
        """Extract text from image using OCR.

        Returns:
            Dict with 'text' string containing extracted text.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        pil_image = Image.fromarray(image)
        text = self._run_inference(pil_image, "ocr")

        return {
            'text': text,
            'model': f"florence-2-{self._model_size}",
        }

    def describe(self, image: np.ndarray) -> dict:
        """Get comprehensive image description including caption and detected objects.

        Returns:
            Dict with 'caption', 'objects', and metadata.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        pil_image = Image.fromarray(image)

        caption = self._run_inference(pil_image, "detailed_caption")
        objects = self._run_inference(pil_image, "object_detection")

        return {
            'caption': caption,
            'objects': objects,
            'model': f"florence-2-{self._model_size}",
        }
