"""PID file management for worker processes."""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Information about a running worker."""

    pid: int
    model_name: str
    socket_path: Path

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        try:
            os.kill(self.pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


class PidStore:
    """Manages PID files for worker processes."""

    def __init__(self, pids_dir: Path):
        self.pids_dir = pids_dir
        self.pids_dir.mkdir(parents=True, exist_ok=True)

    def _pid_file_path(self, model_name: str) -> Path:
        """Get path to PID file for a model."""
        return self.pids_dir / f"{model_name}.pid.json"

    def save(self, info: WorkerInfo) -> None:
        """Save worker info to PID file."""
        path = self._pid_file_path(info.model_name)
        with open(path, "w") as f:
            json.dump(
                {
                    "pid": info.pid,
                    "model_name": info.model_name,
                    "socket_path": str(info.socket_path),
                },
                f,
            )
        logger.debug(f"Saved PID file: {path}")

    def load(self, model_name: str) -> Optional[WorkerInfo]:
        """Load worker info from PID file."""
        path = self._pid_file_path(model_name)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            return WorkerInfo(
                pid=data["pid"],
                model_name=data["model_name"],
                socket_path=Path(data["socket_path"]),
            )
        except Exception as e:
            logger.warning(f"Failed to load PID file {path}: {e}")
            return None

    def remove(self, model_name: str) -> None:
        """Remove PID file for a model."""
        path = self._pid_file_path(model_name)
        if path.exists():
            try:
                path.unlink()
                logger.debug(f"Removed PID file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove PID file {path}: {e}")

    def list_all(self) -> List[WorkerInfo]:
        """List all workers from PID files."""
        workers = []
        for path in self.pids_dir.glob("*.pid.json"):
            model_name = path.stem.replace(".pid", "")
            info = self.load(model_name)
            if info:
                workers.append(info)
        return workers

    def cleanup_stale(self) -> List[str]:
        """Remove PID files for dead processes."""
        removed = []
        for info in self.list_all():
            if not info.is_alive():
                self.remove(info.model_name)
                # Also clean up socket file if exists
                if info.socket_path.exists():
                    try:
                        info.socket_path.unlink()
                    except Exception:
                        pass
                removed.append(info.model_name)
                logger.info(f"Cleaned up stale worker: {info.model_name}")
        return removed
