"""Worker process entry point."""

import os
import sys
import json
import signal
import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from modelstag.config.models import ModelConfig
from modelstag.workers.registry import get_worker_class
from modelstag.workers.base import BaseModelWorker
from modelstag.ipc.protocol import (
    IPCMessage,
    MessageType,
    PingMessage,
    PongMessage,
    ReadyMessage,
    ShutdownMessage,
    InferenceRequest,
    InferenceResponse,
    ErrorMessage,
)
from modelstag.ipc.socket_transport import SocketServer
from modelstag.ipc.image_transfer import ImageTransfer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WorkerProcess:
    """Main worker process controller."""

    def __init__(self, config: ModelConfig, socket_path: Path, pid_file: Path):
        self.config = config
        self.socket_path = socket_path
        self.pid_file = pid_file
        self.worker: Optional[BaseModelWorker] = None
        self.server: Optional[SocketServer] = None
        self.image_transfer = ImageTransfer()
        self._shutdown_event = asyncio.Event()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            self._shutdown_event.set()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

    def _write_pid_file(self):
        """Write PID file."""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pid_file, "w") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "model_name": self.config.name,
                    "socket_path": str(self.socket_path),
                },
                f,
            )
        logger.info(f"Wrote PID file: {self.pid_file}")

    def _remove_pid_file(self):
        """Remove PID file."""
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
                logger.info(f"Removed PID file: {self.pid_file}")
            except Exception as e:
                logger.warning(f"Failed to remove PID file: {e}")

    async def _handle_message(self, message: IPCMessage) -> Optional[IPCMessage]:
        """Handle incoming IPC message."""
        logger.debug(f"Received message: {message.type}")

        if isinstance(message, PingMessage):
            return PongMessage()

        elif isinstance(message, ShutdownMessage):
            logger.info("Received shutdown command")
            self._shutdown_event.set()
            return None

        elif isinstance(message, InferenceRequest):
            return await self._handle_inference(message)

        else:
            logger.warning(f"Unknown message type: {message.type}")
            return ErrorMessage(error=f"Unknown message type: {message.type}")

    async def _handle_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Handle inference request."""
        try:
            if self.worker is None or not self.worker.is_loaded:
                return InferenceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Model not loaded",
                )

            # Read image from shared memory
            image = self.image_transfer.read_image(
                request.shm_name,
                request.image_shape,
                request.image_dtype,
                unlink=True,
            )

            # Run inference
            result = self.worker.process_request(
                image,
                request.output_formats,
                **request.options,
            )

            if result.error:
                return InferenceResponse(
                    request_id=request.request_id,
                    success=False,
                    error=result.error,
                )

            # Write mask to shared memory
            shm_name = None
            mask_shape = ()
            mask_dtype = "uint8"

            if result.mask:
                shm_name, mask_shape, mask_dtype = self.image_transfer.write_image(
                    result.mask.mask
                )

            return InferenceResponse(
                request_id=request.request_id,
                success=True,
                shm_name=shm_name,
                mask_shape=mask_shape,
                mask_dtype=mask_dtype,
                polygons=(
                    result.polygons.polygons if result.polygons else None
                ),
                bbox=result.bbox.to_list() if result.bbox else None,
            )

        except Exception as e:
            logger.exception("Inference error")
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
            )

    async def run(self):
        """Run the worker process."""
        self._setup_signal_handlers()
        self._write_pid_file()

        try:
            # Create and load worker
            worker_class = get_worker_class(self.config.type)
            self.worker = worker_class(self.config)
            self.worker.load_model()

            # Start socket server
            self.server = SocketServer(self.socket_path)
            await self.server.start(self._handle_message)

            logger.info(f"Worker ready: {self.config.name}")

            # Wait for shutdown
            await self._shutdown_event.wait()

        except Exception as e:
            logger.exception(f"Worker error: {e}")
            raise

        finally:
            # Cleanup
            logger.info("Shutting down worker")

            if self.server:
                await self.server.stop()

            if self.worker:
                self.worker.unload_model()

            self.image_transfer.cleanup()
            self._remove_pid_file()

            logger.info("Worker shutdown complete")


def main():
    """Worker process entry point."""
    if len(sys.argv) < 2:
        print("Usage: worker_main.py <config_json>", file=sys.stderr)
        sys.exit(1)

    # Parse config from command line
    config_json = sys.argv[1]
    config_data = json.loads(config_json)

    config = ModelConfig(**config_data["config"])
    socket_path = Path(config_data["socket_path"])
    pid_file = Path(config_data["pid_file"])

    # Run worker
    worker = WorkerProcess(config, socket_path, pid_file)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
