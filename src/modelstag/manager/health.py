"""Health checking for workers."""

import asyncio
import logging
from typing import Optional

from modelstag.ipc.protocol import PingMessage, PongMessage, MessageType
from modelstag.ipc.base_transport import BaseTransport
from modelstag.core.exceptions import IPCError

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checking for worker processes."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    async def check(self, transport: BaseTransport) -> bool:
        """Check if worker is healthy via ping/pong."""
        if not transport.is_connected():
            try:
                await transport.connect()
            except IPCError:
                return False

        try:
            response = await transport.send_and_receive(
                PingMessage(), timeout=self.timeout
            )
            return response.type == MessageType.PONG
        except IPCError as e:
            logger.debug(f"Health check failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected health check error: {e}")
            return False

    async def wait_for_ready(
        self,
        transport: BaseTransport,
        timeout: float = 60.0,
        interval: float = 0.5,
    ) -> bool:
        """Wait for worker to become ready."""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                await transport.connect()
                if await self.check(transport):
                    return True
            except IPCError:
                pass

            await asyncio.sleep(interval)

        return False
