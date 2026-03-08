"""Response schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a single model."""

    name: str
    status: str
    type: str
    startup: str
    output_formats: List[str]
    error: Optional[str] = None


class ModelsListResponse(BaseModel):
    """Response for models list endpoint."""

    models: List[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models: Dict[str, str] = Field(default_factory=dict)


class BBoxResponse(BaseModel):
    """Bounding box in response."""

    x: float
    y: float
    width: float
    height: float


class InferenceResultResponse(BaseModel):
    """Response for inference endpoint."""

    model: str
    processing_time_ms: float
    mask: Optional[dict] = None  # {width, height, data (base64)}
    polygons: Optional[dict] = None  # GeoJSON geometry
    bbox: Optional[BBoxResponse] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
