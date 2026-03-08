"""Converters for mask to other output formats."""

import io
import cv2
import numpy as np
from PIL import Image

from modelstag.outputs.base import PolygonOutput, BBoxOutput


def mask_to_polygons(
    mask: np.ndarray,
    simplify_epsilon: float = 1.0,
    min_area: float = 100.0,
) -> PolygonOutput:
    """Convert binary mask to polygons using OpenCV contour detection.

    Args:
        mask: Binary mask (uint8, 0-255)
        simplify_epsilon: Epsilon for Douglas-Peucker simplification
        min_area: Minimum contour area to include

    Returns:
        PolygonOutput with list of polygons
    """
    # Ensure binary mask
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Threshold to ensure binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Simplify contour
        if simplify_epsilon > 0:
            contour = cv2.approxPolyDP(contour, simplify_epsilon, True)

        # Convert to list of [x, y] points
        points = contour.squeeze().tolist()
        if isinstance(points[0], (int, float)):
            # Single point, skip
            continue

        # Ensure polygon is closed
        if points[0] != points[-1]:
            points.append(points[0])

        polygons.append(points)

    return PolygonOutput(polygons=polygons)


def mask_to_bbox(mask: np.ndarray) -> BBoxOutput:
    """Convert binary mask to bounding box.

    Args:
        mask: Binary mask (uint8, 0-255)

    Returns:
        BBoxOutput with bounding box coordinates
    """
    # Find non-zero pixels
    coords = cv2.findNonZero(mask)

    if coords is None:
        # No foreground pixels
        return BBoxOutput(x=0, y=0, width=0, height=0)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)

    return BBoxOutput(x=float(x), y=float(y), width=float(w), height=float(h))


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Convert mask to PNG bytes.

    Args:
        mask: Mask array (uint8)

    Returns:
        PNG image as bytes
    """
    # Ensure 2D grayscale array
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D grayscale array")
    img = Image.fromarray(mask)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
