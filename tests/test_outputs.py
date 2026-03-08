"""Tests for output converters."""

import pytest
import numpy as np

from modelstag.outputs.base import MaskOutput, PolygonOutput, BBoxOutput, InferenceResult
from modelstag.outputs.converters import mask_to_polygons, mask_to_bbox, mask_to_png_bytes


def test_mask_output_from_array():
    """Test creating MaskOutput from numpy array."""
    arr = np.zeros((100, 200), dtype=np.uint8)
    arr[20:80, 40:160] = 255

    mask = MaskOutput.from_array(arr)

    assert mask.height == 100
    assert mask.width == 200
    assert mask.mask.shape == (100, 200)


def test_mask_output_from_rgb_array():
    """Test creating MaskOutput from RGB array."""
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    arr[20:80, 40:160, :] = 255

    mask = MaskOutput.from_array(arr)

    assert mask.height == 100
    assert mask.width == 200
    assert mask.mask.shape == (100, 200)


def test_polygon_output_to_geojson_single():
    """Test single polygon to GeoJSON."""
    poly = PolygonOutput(polygons=[[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]])
    geojson = poly.to_geojson()

    assert geojson["type"] == "Polygon"
    assert len(geojson["coordinates"]) == 1


def test_polygon_output_to_geojson_multi():
    """Test multiple polygons to GeoJSON."""
    poly = PolygonOutput(
        polygons=[
            [[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]],
            [[60, 60], [100, 60], [100, 100], [60, 100], [60, 60]],
        ]
    )
    geojson = poly.to_geojson()

    assert geojson["type"] == "MultiPolygon"
    assert len(geojson["coordinates"]) == 2


def test_bbox_output_to_dict():
    """Test BBoxOutput to dictionary."""
    bbox = BBoxOutput(x=10, y=20, width=80, height=60)
    d = bbox.to_dict()

    assert d == {"x": 10, "y": 20, "width": 80, "height": 60}


def test_mask_to_polygons():
    """Test converting mask to polygons."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    result = mask_to_polygons(mask)

    assert isinstance(result, PolygonOutput)
    assert len(result.polygons) >= 1


def test_mask_to_bbox():
    """Test converting mask to bounding box."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:90] = 255

    result = mask_to_bbox(mask)

    assert isinstance(result, BBoxOutput)
    assert result.x == 30
    assert result.y == 20
    assert result.width == 60
    assert result.height == 60


def test_mask_to_bbox_empty():
    """Test converting empty mask to bounding box."""
    mask = np.zeros((100, 100), dtype=np.uint8)

    result = mask_to_bbox(mask)

    assert result.x == 0
    assert result.y == 0
    assert result.width == 0
    assert result.height == 0


def test_mask_to_png_bytes():
    """Test converting mask to PNG bytes."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    png_bytes = mask_to_png_bytes(mask)

    # Check PNG magic bytes
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_inference_result_to_dict():
    """Test InferenceResult to dictionary."""
    result = InferenceResult(
        model_name="test",
        mask=MaskOutput(np.zeros((100, 100), dtype=np.uint8), 100, 100),
        bbox=BBoxOutput(x=10, y=20, width=80, height=60),
        processing_time_ms=123.45,
    )

    d = result.to_dict()

    assert d["model"] == "test"
    assert d["processing_time_ms"] == 123.45
    assert d["mask"]["width"] == 100
    assert d["bbox"]["x"] == 10
