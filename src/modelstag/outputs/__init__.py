"""Output types and converters."""

from modelstag.outputs.base import MaskOutput, PolygonOutput, BBoxOutput, InferenceResult
from modelstag.outputs.converters import mask_to_polygons, mask_to_bbox, mask_to_png_bytes

__all__ = [
    "MaskOutput",
    "PolygonOutput",
    "BBoxOutput",
    "InferenceResult",
    "mask_to_polygons",
    "mask_to_bbox",
    "mask_to_png_bytes",
]
