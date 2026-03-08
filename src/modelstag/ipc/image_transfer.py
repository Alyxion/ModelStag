"""Shared memory image transfer for zero-copy IPC."""

import uuid
import logging
from multiprocessing import shared_memory
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class SharedImageBuffer:
    """A single shared memory buffer for image data."""

    def __init__(self, name: str, size: int, create: bool = True):
        self.name = name
        self.size = size
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._create = create

    def open(self) -> None:
        """Open or create the shared memory buffer."""
        if self._shm is not None:
            return

        try:
            if self._create:
                self._shm = shared_memory.SharedMemory(
                    name=self.name, create=True, size=self.size
                )
            else:
                self._shm = shared_memory.SharedMemory(name=self.name, create=False)
        except FileExistsError:
            # Already exists, open it
            self._shm = shared_memory.SharedMemory(name=self.name, create=False)

    def close(self) -> None:
        """Close the shared memory buffer."""
        if self._shm:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None

    def unlink(self) -> None:
        """Unlink (delete) the shared memory buffer."""
        if self._shm:
            try:
                self._shm.unlink()
            except Exception:
                pass

    def write_array(self, arr: np.ndarray) -> None:
        """Write numpy array to shared memory."""
        if not self._shm:
            raise RuntimeError("Buffer not open")

        flat = arr.tobytes()
        if len(flat) > self.size:
            raise ValueError(f"Array too large: {len(flat)} > {self.size}")

        self._shm.buf[: len(flat)] = flat

    def read_array(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Read numpy array from shared memory."""
        if not self._shm:
            raise RuntimeError("Buffer not open")

        arr = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)
        return arr.copy()  # Copy to detach from shared memory


class ImageTransfer:
    """Manages shared memory buffers for image transfer."""

    def __init__(self):
        self._buffers: dict[str, SharedImageBuffer] = {}

    def create_buffer(self, size: int) -> str:
        """Create a new shared memory buffer."""
        name = f"modelstag_{uuid.uuid4().hex[:12]}"
        buf = SharedImageBuffer(name, size, create=True)
        buf.open()
        self._buffers[name] = buf
        logger.debug(f"Created shared memory buffer: {name} ({size} bytes)")
        return name

    def write_image(self, image: np.ndarray) -> tuple[str, tuple, str]:
        """Write image to shared memory, returns (name, shape, dtype)."""
        size = image.nbytes
        name = self.create_buffer(size)
        self._buffers[name].write_array(image)
        return name, image.shape, str(image.dtype)

    def read_image(
        self, name: str, shape: tuple, dtype: str, unlink: bool = True
    ) -> np.ndarray:
        """Read image from shared memory."""
        if name in self._buffers:
            buf = self._buffers[name]
        else:
            buf = SharedImageBuffer(name, 0, create=False)
            buf.open()
            self._buffers[name] = buf

        arr = buf.read_array(shape, np.dtype(dtype))

        if unlink:
            self.release_buffer(name)

        return arr

    def release_buffer(self, name: str) -> None:
        """Release a shared memory buffer."""
        if name in self._buffers:
            buf = self._buffers.pop(name)
            buf.close()
            buf.unlink()
            logger.debug(f"Released shared memory buffer: {name}")

    def cleanup(self) -> None:
        """Clean up all buffers."""
        for name in list(self._buffers.keys()):
            self.release_buffer(name)
