"""Tests for configuration loading."""

import pytest
from pathlib import Path

from modelstag.core.types import ModelType, StartupMode, ProcessMode
from modelstag.config.models import ModelConfig
from modelstag.config.settings import Settings


def test_model_config_from_toml():
    """Test creating ModelConfig from TOML data."""
    data = {
        "type": "rembg",
        "enabled": True,
        "startup": "eager",
        "rembg_model": "u2net",
        "output_formats": ["mask", "polygon"],
    }

    config = ModelConfig.from_toml("test_model", data)

    assert config.name == "test_model"
    assert config.type == ModelType.REMBG
    assert config.enabled is True
    assert config.startup == StartupMode.EAGER
    assert config.rembg_model == "u2net"
    assert config.output_formats == ["mask", "polygon"]


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig(name="test")

    assert config.type == ModelType.REMBG
    assert config.enabled is True
    assert config.startup == StartupMode.LAZY
    assert config.rembg_model is None
    assert "mask" in config.output_formats


def test_settings_paths():
    """Test Settings path properties."""
    settings = Settings(run_dir=Path("/tmp/test_modelstag"))

    assert settings.pids_dir == Path("/tmp/test_modelstag/pids")
    assert settings.sockets_dir == Path("/tmp/test_modelstag/sockets")
    assert settings.logs_dir == Path("/tmp/test_modelstag/logs")
