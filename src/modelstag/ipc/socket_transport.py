"""Unix Domain Socket transport for IPC."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Awaitable

from modelstag.ipc.protocol import IPCMessage
from modelstag.ipc.base_transport import BaseTransport
from modelstag.core.exceptions import IPCError

logger = logging.getLogger(__name__)


class SocketTransport(BaseTransport):
    """Unix Domain Socket client transport."""

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to the Unix socket."""
        if self._connected:
            return

        if not self.socket_path.exists():
            raise IPCError(f"Socket file not found: {self.socket_path}")

        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                str(self.socket_path)
            )
            self._connected = True
            logger.debug(f"Connected to socket: {self.socket_path}")
        except Exception as e:
            raise IPCError(f"Failed to connect to socket: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from the socket."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self._connected = False
        logger.debug(f"Disconnected from socket: {self.socket_path}")

    async def send(self, message: IPCMessage) -> None:
        """Send a message over the socket."""
        if not self._connected or not self._writer:
            raise IPCError("Not connected")

        async with self._lock:
            try:
                data = message.to_json().encode("utf-8")
                # Send length prefix (4 bytes) + data
                length = len(data)
                self._writer.write(length.to_bytes(4, "big") + data)
                await self._writer.drain()
            except Exception as e:
                self._connected = False
                raise IPCError(f"Failed to send message: {e}") from e

    async def receive(self, timeout: Optional[float] = None) -> IPCMessage:
        """Receive a message from the socket."""
        if not self._connected or not self._reader:
            raise IPCError("Not connected")

        try:
            # Read length prefix
            coro = self._reader.readexactly(4)
            if timeout:
                length_bytes = await asyncio.wait_for(coro, timeout)
            else:
                length_bytes = await coro

            length = int.from_bytes(length_bytes, "big")

            # Read message data
            coro = self._reader.readexactly(length)
            if timeout:
                data = await asyncio.wait_for(coro, timeout)
            else:
                data = await coro

            return IPCMessage.from_json(data.decode("utf-8"))
        except asyncio.TimeoutError:
            raise IPCError("Receive timeout")
        except asyncio.IncompleteReadError:
            self._connected = False
            raise IPCError("Connection closed")
        except Exception as e:
            self._connected = False
            raise IPCError(f"Failed to receive message: {e}") from e

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


class SocketServer:
    """Unix Domain Socket server for worker process."""

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path
        self._server: Optional[asyncio.Server] = None
        self._running = False

    async def start(
        self,
        handler: Callable[[IPCMessage], Awaitable[Optional[IPCMessage]]],
    ) -> None:
        """Start the socket server."""
        # Clean up existing socket file
        if self.socket_path.exists():
            self.socket_path.unlink()

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        async def client_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ):
            """Handle a client connection."""
            logger.debug("Client connected")
            try:
                while self._running:
                    try:
                        # Read length prefix
                        length_bytes = await reader.readexactly(4)
                        length = int.from_bytes(length_bytes, "big")

                        # Read message
                        data = await reader.readexactly(length)
                        message = IPCMessage.from_json(data.decode("utf-8"))

                        # Handle message
                        response = await handler(message)

                        if response:
                            # Send response
                            response_data = response.to_json().encode("utf-8")
                            writer.write(len(response_data).to_bytes(4, "big"))
                            writer.write(response_data)
                            await writer.drain()

                    except asyncio.IncompleteReadError:
                        logger.debug("Client disconnected")
                        break
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                        break
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        self._server = await asyncio.start_unix_server(
            client_handler, str(self.socket_path)
        )
        self._running = True
        logger.info(f"Socket server started: {self.socket_path}")

    async def stop(self) -> None:
        """Stop the socket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Clean up socket file
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception:
                pass

        logger.info("Socket server stopped")

    async def serve_forever(self) -> None:
        """Run server until stopped."""
        if self._server:
            await self._server.serve_forever()
