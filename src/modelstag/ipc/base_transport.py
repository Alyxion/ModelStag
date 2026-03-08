"""Abstract transport interface for IPC."""

from abc import ABC, abstractmethod
from typing import Optional

from modelstag.ipc.protocol import IPCMessage


class BaseTransport(ABC):
    """Abstract base class for IPC transport."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def send(self, message: IPCMessage) -> None:
        """Send a message."""
        pass

    @abstractmethod
    async def receive(self, timeout: Optional[float] = None) -> IPCMessage:
        """Receive a message."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        pass

    async def send_and_receive(
        self, message: IPCMessage, timeout: Optional[float] = None
    ) -> IPCMessage:
        """Send a message and wait for response."""
        await self.send(message)
        return await self.receive(timeout)
