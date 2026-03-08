"""Process manager for worker lifecycle - synchronous with threading.

NOTE: With multiple uvicorn workers, each worker has its own ProcessManager.
Models loaded in one worker are NOT available in other workers.
Use --workers 1 for consistent model state, or implement external model serving.
"""

import os
import sys
import json
import signal
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

from modelstag.config.settings import Settings
from modelstag.config.models import ModelConfig
from modelstag.core.types import ModelStatus, StartupMode, ProcessMode
from modelstag.core.exceptions import (
    ModelNotFoundError,
    ModelNotRunningError,
    ModelAlreadyRunningError,
)
from modelstag.workers.base import BaseModelWorker
from modelstag.workers.registry import get_worker_class
from modelstag.outputs.base import InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    """Runtime state for a worker."""
    config: ModelConfig
    status: ModelStatus = ModelStatus.STOPPED
    process: Optional[subprocess.Popen] = None
    direct_worker: Optional[BaseModelWorker] = None
    error: Optional[str] = None
    lock: threading.Lock = None

    def __post_init__(self):
        self.lock = threading.Lock()


class ProcessManager:
    """Manages worker process lifecycle - fully synchronous."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._workers: Dict[str, WorkerState] = {}
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="modelstag")
        self._request_counter = 0

    def startup(self) -> None:
        """Initialize manager and start eager workers."""
        logger.info("Starting ProcessManager")

        # Initialize worker states
        for config in self.settings.get_enabled_models():
            self._workers[config.name] = WorkerState(config=config)

        logger.info("ProcessManager startup complete")

    def shutdown(self) -> None:
        """Shutdown manager."""
        logger.info("Shutting down ProcessManager")

        # Unload all workers
        for name, state in self._workers.items():
            with state.lock:
                if state.direct_worker:
                    try:
                        state.direct_worker.unload_model()
                    except Exception:
                        pass
                    state.direct_worker = None
                    state.status = ModelStatus.STOPPED

        self._executor.shutdown(wait=False)
        logger.info("ProcessManager shutdown complete")

    def start_worker(self, model_name: str) -> None:
        """Start a worker (non-blocking - runs in background thread)."""
        if model_name not in self._workers:
            raise ModelNotFoundError(model_name)

        state = self._workers[model_name]

        with state.lock:
            if state.status == ModelStatus.RUNNING:
                raise ModelAlreadyRunningError(model_name)
            if state.status == ModelStatus.STARTING:
                return  # Already starting

            state.status = ModelStatus.STARTING
            state.error = None

        # Start loading in background thread
        self._executor.submit(self._load_worker_sync, model_name)

    def _load_worker_sync(self, model_name: str) -> None:
        """Load worker synchronously (runs in thread pool)."""
        state = self._workers[model_name]

        try:
            worker_class = get_worker_class(state.config.type)
            worker = worker_class(state.config)
            worker.load_model()

            with state.lock:
                state.direct_worker = worker
                state.status = ModelStatus.RUNNING
                logger.info(f"Worker started: {model_name}")

        except Exception as e:
            with state.lock:
                state.status = ModelStatus.ERROR
                state.error = str(e)
            logger.error(f"Failed to start worker {model_name}: {e}")

    def stop_worker(self, model_name: str) -> None:
        """Stop a worker."""
        if model_name not in self._workers:
            raise ModelNotFoundError(model_name)

        state = self._workers[model_name]

        with state.lock:
            if state.status != ModelStatus.RUNNING:
                raise ModelNotRunningError(model_name)

            state.status = ModelStatus.STOPPING

            if state.direct_worker:
                try:
                    state.direct_worker.unload_model()
                except Exception as e:
                    logger.error(f"Error unloading worker {model_name}: {e}")
                state.direct_worker = None

            state.status = ModelStatus.STOPPED
            logger.info(f"Worker stopped: {model_name}")

    def stop_all(self) -> None:
        """Stop all running workers."""
        for name, state in self._workers.items():
            if state.status == ModelStatus.RUNNING:
                try:
                    self.stop_worker(name)
                except Exception as e:
                    logger.error(f"Error stopping worker {name}: {e}")

    def run_inference(
        self,
        model_name: str,
        image: np.ndarray,
        output_formats: List[str],
        **options,
    ) -> InferenceResult:
        """Run inference on a model."""
        if model_name not in self._workers:
            raise ModelNotFoundError(model_name)

        state = self._workers[model_name]

        # Check status
        if state.status == ModelStatus.STOPPED:
            if state.config.startup == StartupMode.LAZY:
                # Start and wait
                self.start_worker(model_name)
                # Wait for loading to complete
                import time
                timeout = 300
                waited = 0
                while state.status == ModelStatus.STARTING and waited < timeout:
                    time.sleep(0.1)
                    waited += 0.1
            else:
                raise ModelNotRunningError(model_name)

        if state.status == ModelStatus.STARTING:
            # Wait for loading
            import time
            timeout = 300
            waited = 0
            while state.status == ModelStatus.STARTING and waited < timeout:
                time.sleep(0.1)
                waited += 0.1

        if state.status != ModelStatus.RUNNING:
            raise ModelNotRunningError(model_name)

        if not state.direct_worker:
            raise ModelNotRunningError(model_name)

        # Run inference (direct call - caller can use thread pool if needed)
        return state.direct_worker.process_request(image, output_formats, **options)

    def get_worker_status(self, model_name: str) -> ModelStatus:
        """Get status of a worker."""
        if model_name not in self._workers:
            raise ModelNotFoundError(model_name)
        return self._workers[model_name].status

    def get_all_statuses(self) -> Dict[str, Dict]:
        """Get status of all workers."""
        return {
            name: {
                "status": state.status.value,
                "type": state.config.type.value,
                "startup": state.config.startup.value,
                "error": state.error,
            }
            for name, state in self._workers.items()
        }
