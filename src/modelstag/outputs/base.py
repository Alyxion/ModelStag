"""Output data structures."""

import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from PIL import Image


@dataclass
class MaskOutput:
    """Binary or segmentation mask output."""

    mask: np.ndarray  # uint8, 0-255 (grayscale HxW or RGB HxWx3)
    width: int
    height: int
    is_segmentation: bool = False  # True if RGB segmentation map

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "MaskOutput":
        """Create from numpy array."""
        is_seg = False
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                # RGB - keep as segmentation map
                is_seg = True
            elif arr.shape[2] == 4:
                arr = arr[:, :, 3]  # Use alpha channel
        return cls(mask=arr, height=arr.shape[0], width=arr.shape[1], is_segmentation=is_seg)

    def to_grayscale(self) -> np.ndarray:
        """Get grayscale version of mask (for polygon/bbox extraction)."""
        if self.is_segmentation:
            # Combine RGB channels - any non-zero becomes white
            return np.any(self.mask > 0, axis=2).astype(np.uint8) * 255
        return self.mask

    def to_base64(self) -> str:
        """Convert mask to base64-encoded PNG."""
        if self.is_segmentation:
            img = Image.fromarray(self.mask, mode='RGB')
        else:
            img = Image.fromarray(self.mask)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class PolygonOutput:
    """Polygon output in GeoJSON-like format."""

    polygons: List[List[List[float]]]  # List of polygons, each is list of [x, y] points

    def to_geojson(self) -> Dict:
        """Convert to GeoJSON geometry."""
        if len(self.polygons) == 1:
            return {
                "type": "Polygon",
                "coordinates": [self.polygons[0]],
            }
        return {
            "type": "MultiPolygon",
            "coordinates": [[poly] for poly in self.polygons],
        }


@dataclass
class BBoxOutput:
    """Bounding box output."""

    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    def to_list(self) -> List[float]:
        """Convert to [x, y, width, height] list."""
        return [self.x, self.y, self.width, self.height]


@dataclass
class InferenceResult:
    """Combined inference result with all output formats."""

    model_name: str
    mask: Optional[MaskOutput] = None
    polygons: Optional[PolygonOutput] = None
    bbox: Optional[BBoxOutput] = None
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON response."""
        result = {
            "model": self.model_name,
            "processing_time_ms": self.processing_time_ms,
        }

        if self.error:
            result["error"] = self.error
            return result

        if self.mask:
            result["mask"] = {
                "width": self.mask.width,
                "height": self.mask.height,
            }

        if self.polygons:
            result["polygons"] = self.polygons.to_geojson()

        if self.bbox:
            result["bbox"] = self.bbox.to_dict()

        return result
