"""IPC protocol message definitions."""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


class MessageType(str, Enum):
    """Types of IPC messages."""

    PING = "ping"
    PONG = "pong"
    READY = "ready"
    SHUTDOWN = "shutdown"
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    ERROR = "error"


@dataclass
class IPCMessage:
    """Base IPC message."""

    type: MessageType

    def to_json(self) -> str:
        """Serialize message to JSON."""
        data = asdict(self)
        data["type"] = self.type.value
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "IPCMessage":
        """Deserialize message from JSON."""
        parsed = json.loads(data)
        msg_type = MessageType(parsed["type"])

        type_map = {
            MessageType.PING: PingMessage,
            MessageType.PONG: PongMessage,
            MessageType.READY: ReadyMessage,
            MessageType.SHUTDOWN: ShutdownMessage,
            MessageType.INFERENCE_REQUEST: InferenceRequest,
            MessageType.INFERENCE_RESPONSE: InferenceResponse,
            MessageType.ERROR: ErrorMessage,
        }

        msg_class = type_map.get(msg_type)
        if msg_class is None:
            raise ValueError(f"Unknown message type: {msg_type}")

        del parsed["type"]
        return msg_class(**parsed)


@dataclass
class PingMessage(IPCMessage):
    """Ping message for health check."""

    type: MessageType = field(default=MessageType.PING, init=False)


@dataclass
class PongMessage(IPCMessage):
    """Pong response to ping."""

    type: MessageType = field(default=MessageType.PONG, init=False)


@dataclass
class ReadyMessage(IPCMessage):
    """Worker ready signal."""

    type: MessageType = field(default=MessageType.READY, init=False)
    model_name: str = ""


@dataclass
class ShutdownMessage(IPCMessage):
    """Shutdown request."""

    type: MessageType = field(default=MessageType.SHUTDOWN, init=False)


@dataclass
class InferenceRequest(IPCMessage):
    """Inference request message."""

    type: MessageType = field(default=MessageType.INFERENCE_REQUEST, init=False)
    request_id: str = ""
    shm_name: Optional[str] = None  # SharedMemory name for image data
    image_shape: Tuple[int, ...] = ()  # (height, width, channels)
    image_dtype: str = "uint8"
    output_formats: List[str] = field(default_factory=lambda: ["mask"])
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse(IPCMessage):
    """Inference response message."""

    type: MessageType = field(default=MessageType.INFERENCE_RESPONSE, init=False)
    request_id: str = ""
    success: bool = True
    shm_name: Optional[str] = None  # SharedMemory name for mask data
    mask_shape: Tuple[int, ...] = ()  # (height, width) or (height, width, channels)
    mask_dtype: str = "uint8"
    polygons: Optional[List[List[List[float]]]] = None  # List of polygons
    bbox: Optional[List[float]] = None  # [x, y, width, height]
    error: Optional[str] = None


@dataclass
class ErrorMessage(IPCMessage):
    """Error message."""

    type: MessageType = field(default=MessageType.ERROR, init=False)
    error: str = ""
    request_id: Optional[str] = None
