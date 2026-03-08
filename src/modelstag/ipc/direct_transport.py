"""Direct in-process transport for single-process mode."""

import asyncio
from typing import Optional, Callable, Awaitable

from modelstag.ipc.protocol import IPCMessage
from modelstag.ipc.base_transport import BaseTransport
from modelstag.core.exceptions import IPCError


class DirectTransport(BaseTransport):
    """Direct in-process transport using async queues."""

    def __init__(
        self,
        handler: Optional[Callable[[IPCMessage], Awaitable[Optional[IPCMessage]]]] = None,
    ):
        self._handler = handler
        self._connected = False
        self._response_queue: asyncio.Queue[IPCMessage] = asyncio.Queue()

    def set_handler(
        self, handler: Callable[[IPCMessage], Awaitable[Optional[IPCMessage]]]
    ) -> None:
        """Set the message handler."""
        self._handler = handler

    async def connect(self) -> None:
        """Mark as connected."""
        if self._handler is None:
            raise IPCError("No handler set")
        self._connected = True

    async def disconnect(self) -> None:
        """Mark as disconnected."""
        self._connected = False

    async def send(self, message: IPCMessage) -> None:
        """Send message directly to handler."""
        if not self._connected:
            raise IPCError("Not connected")
        if not self._handler:
            raise IPCError("No handler set")

        # Call handler directly and queue response
        response = await self._handler(message)
        if response:
            await self._response_queue.put(response)

    async def receive(self, timeout: Optional[float] = None) -> IPCMessage:
        """Receive response from queue."""
        if not self._connected:
            raise IPCError("Not connected")

        try:
            if timeout:
                return await asyncio.wait_for(self._response_queue.get(), timeout)
            else:
                return await self._response_queue.get()
        except asyncio.TimeoutError:
            raise IPCError("Receive timeout")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
