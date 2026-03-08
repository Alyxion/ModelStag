"""Model configuration schema."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from modelstag.core.types import ModelType, StartupMode


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    name: str = Field(description="Model identifier")
    type: ModelType = Field(default=ModelType.REMBG, description="Worker type")
    enabled: bool = Field(default=True, description="Whether model is available")
    startup: StartupMode = Field(
        default=StartupMode.LAZY, description="When to start worker"
    )
    rembg_model: Optional[str] = Field(
        default=None, description="Rembg model name (for rembg type)"
    )
    output_formats: List[str] = Field(
        default_factory=lambda: ["mask", "polygon", "bbox"],
        description="Supported output formats",
    )
    env: Dict[str, str] = Field(
        default_factory=dict, description="Environment variable overrides"
    )

    model_config = {"extra": "allow"}

    @classmethod
    def from_toml(cls, name: str, data: dict) -> "ModelConfig":
        """Create ModelConfig from TOML data."""
        return cls(name=name, **data)
