"""Tests for IPC protocol."""

import pytest

from modelstag.ipc.protocol import (
    MessageType,
    IPCMessage,
    PingMessage,
    PongMessage,
    InferenceRequest,
    InferenceResponse,
)


def test_ping_message_serialization():
    """Test ping message JSON serialization."""
    msg = PingMessage()
    json_str = msg.to_json()
    parsed = IPCMessage.from_json(json_str)

    assert isinstance(parsed, PingMessage)
    assert parsed.type == MessageType.PING


def test_pong_message_serialization():
    """Test pong message JSON serialization."""
    msg = PongMessage()
    json_str = msg.to_json()
    parsed = IPCMessage.from_json(json_str)

    assert isinstance(parsed, PongMessage)
    assert parsed.type == MessageType.PONG


def test_inference_request_serialization():
    """Test inference request JSON serialization."""
    msg = InferenceRequest(
        request_id="req_123",
        shm_name="test_shm",
        image_shape=(480, 640, 3),
        image_dtype="uint8",
        output_formats=["mask", "polygon"],
        options={"alpha_matting": True},
    )

    json_str = msg.to_json()
    parsed = IPCMessage.from_json(json_str)

    assert isinstance(parsed, InferenceRequest)
    assert parsed.request_id == "req_123"
    assert parsed.shm_name == "test_shm"
    assert parsed.image_shape == [480, 640, 3]  # JSON converts to list
    assert parsed.output_formats == ["mask", "polygon"]
    assert parsed.options == {"alpha_matting": True}


def test_inference_response_serialization():
    """Test inference response JSON serialization."""
    msg = InferenceResponse(
        request_id="req_123",
        success=True,
        shm_name="mask_shm",
        mask_shape=(480, 640),
        polygons=[[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]],
        bbox=[10, 20, 80, 60],
    )

    json_str = msg.to_json()
    parsed = IPCMessage.from_json(json_str)

    assert isinstance(parsed, InferenceResponse)
    assert parsed.request_id == "req_123"
    assert parsed.success is True
    assert parsed.polygons is not None
    assert len(parsed.polygons) == 1
    assert parsed.bbox == [10, 20, 80, 60]
