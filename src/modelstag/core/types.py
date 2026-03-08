"""Shared type definitions."""

from enum import Enum


class ModelStatus(str, Enum):
    """Status of a model worker."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    STOPPING = "stopping"


class StartupMode(str, Enum):
    """When to start a model worker."""

    EAGER = "eager"  # Start on server startup
    LAZY = "lazy"    # Start on first request
    MANUAL = "manual"  # Only start via API


class ModelType(str, Enum):
    """Type of model worker."""

    REMBG = "rembg"
    SAM = "sam"  # Segment Anything Model
    DEPTH = "depth"  # Depth estimation (Depth Anything)
    DETECT = "detect"  # Object detection (YOLO-World)
    CAPTION = "caption"  # Image captioning (Florence-2)
    POSE = "pose"  # Pose estimation (RTMO, RTMPose, MediaPipe)
    HAND = "hand"  # Hand tracking (HaMeR, MediaPipe Hands)
    CUSTOM_ONNX = "custom_onnx"


class ProcessMode(str, Enum):
    """Process execution mode."""

    MULTI = "multi"   # Each model in its own process
    SINGLE = "single"  # All models in FastAPI process
