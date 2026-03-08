"""Application settings."""

import sys
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from modelstag.core.types import ProcessMode
from modelstag.config.models import ModelConfig


class Settings(BaseSettings):
    """Application settings loaded from environment and config file."""

    # Paths
    config_path: Path = Field(
        default=Path("config/models.toml"),
        description="Path to models configuration file",
    )
    run_dir: Path = Field(
        default=Path("run"), description="Runtime directory for sockets, pids, logs"
    )

    # Process settings
    process_mode: ProcessMode = Field(
        default=ProcessMode.MULTI, description="Process execution mode"
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # Timeouts
    worker_startup_timeout: float = Field(
        default=60.0, description="Timeout for worker startup"
    )
    worker_shutdown_timeout: float = Field(
        default=10.0, description="Timeout for worker shutdown"
    )
    inference_timeout: float = Field(default=30.0, description="Timeout for inference")

    # Loaded models (populated from config file)
    _models: Dict[str, ModelConfig] = {}

    model_config = {"env_prefix": "MODELSTAG_", "extra": "ignore"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_config()

    def _load_config(self) -> None:
        """Load model configurations from TOML file."""
        if not self.config_path.exists():
            return

        with open(self.config_path, "rb") as f:
            data = tomllib.load(f)

        # Load process mode from config only if not set via env
        import os
        if "process" in data and "mode" in data["process"]:
            if "MODELSTAG_PROCESS_MODE" not in os.environ:
                self.process_mode = ProcessMode(data["process"]["mode"])

        # Load model configurations
        models_data = data.get("models", {})
        for name, model_data in models_data.items():
            self._models[name] = ModelConfig.from_toml(name, model_data)

    @property
    def models(self) -> Dict[str, ModelConfig]:
        """Get all model configurations."""
        return self._models

    @property
    def pids_dir(self) -> Path:
        """Directory for PID files."""
        path = self.run_dir / "pids"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sockets_dir(self) -> Path:
        """Directory for socket files."""
        path = self.run_dir / "sockets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        path = self.run_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name."""
        return self._models.get(name)

    def get_enabled_models(self) -> List[ModelConfig]:
        """Get all enabled model configurations."""
        return [m for m in self._models.values() if m.enabled]

    def get_eager_models(self) -> List[ModelConfig]:
        """Get models that should start eagerly."""
        from modelstag.core.types import StartupMode

        return [
            m
            for m in self._models.values()
            if m.enabled and m.startup == StartupMode.EAGER
        ]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
