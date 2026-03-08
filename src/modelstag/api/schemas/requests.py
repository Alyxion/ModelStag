"""Request schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class InferenceOptions(BaseModel):
    """Options for inference request."""

    output_formats: List[str] = Field(
        default=["mask", "polygon", "bbox"],
        description="Desired output formats",
    )
    alpha_matting: bool = Field(
        default=False,
        description="Use alpha matting for better edges",
    )
    alpha_matting_foreground_threshold: int = Field(
        default=240,
        description="Foreground threshold for alpha matting",
    )
    alpha_matting_background_threshold: int = Field(
        default=10,
        description="Background threshold for alpha matting",
    )
