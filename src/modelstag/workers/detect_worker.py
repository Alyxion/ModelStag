"""YOLO-World object detection worker."""

import logging
from typing import List, Optional

import numpy as np
from PIL import Image

from modelstag.core.types import ModelType
from modelstag.config.models import ModelConfig
from modelstag.workers.base import BaseModelWorker
from modelstag.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)

YOLO_MODELS = {
    "small": "yolov8s-worldv2",
    "medium": "yolov8m-worldv2",
    "large": "yolov8l-worldv2",
    "xlarge": "yolov8x-worldv2",
}


@WorkerRegistry.register(ModelType.DETECT)
class DetectWorker(BaseModelWorker):
    """Worker for YOLO-World open-vocabulary object detection."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._model_size = getattr(config, 'detect_model', 'small') or 'small'
        self._default_classes = getattr(config, 'detect_classes', None) or [
            "person", "car", "dog", "cat", "bird", "chair", "table",
            "phone", "laptop", "bottle", "cup", "book", "bag"
        ]

    def load_model(self) -> None:
        """Load the YOLO-World model."""
        if self._loaded:
            return

        from ultralytics import YOLO

        logger.info(f"Loading YOLO-World model: {self._model_size}")

        model_name = YOLO_MODELS.get(self._model_size, YOLO_MODELS["small"])
        self._model = YOLO(f"{model_name}.pt")

        # Set default classes
        self._model.set_classes(self._default_classes)

        self._loaded = True
        logger.info(f"Loaded YOLO-World model: {self._model_size}")

    def unload_model(self) -> None:
        """Unload the model."""
        if not self._loaded:
            return

        logger.info(f"Unloading YOLO-World model: {self._model_size}")
        self._model = None
        self._loaded = False

    def predict(self, image: np.ndarray, **options) -> np.ndarray:
        """Run object detection and return a labeled mask.

        Each detected object gets a unique ID in the mask.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        # Set custom classes if provided
        classes = options.get('classes', None)
        if classes:
            self._model.set_classes(classes)

        # Run detection
        results = self._model.predict(image, verbose=False)

        # Create labeled mask
        h, w = image.shape[:2]
        labeled = np.zeros((h, w), dtype=np.uint8)

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Fill box region with label ID
                labeled[y1:y2, x1:x2] = min(i + 1, 255)

        return labeled

    def predict_boxes(self, image: np.ndarray, classes: List[str] = None,
                     confidence: float = 0.25) -> dict:
        """Run object detection and return bounding boxes with labels.

        Args:
            image: Input image as numpy array
            classes: List of class names to detect (uses defaults if None)
            confidence: Minimum confidence threshold

        Returns:
            Dict with 'detections' list containing boxes, labels, and scores.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        # Set custom classes if provided
        if classes:
            self._model.set_classes(classes)
        else:
            self._model.set_classes(self._default_classes)

        # Run detection
        results = self._model.predict(image, conf=confidence, verbose=False)

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                detections.append({
                    'bbox': {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                    },
                    'label': names.get(cls_id, f"class_{cls_id}"),
                    'confidence': conf,
                })

        return {
            'detections': detections,
            'count': len(detections),
            'classes_searched': classes or self._default_classes,
        }
