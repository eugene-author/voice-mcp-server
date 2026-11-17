"""Unloads models that stayed idle inside a running process, freeing GPU memory without stopping it."""

import asyncio
import logging

from app.services.model_host import ModelHost

logger = logging.getLogger(__name__)
MAX_CHECK_SECONDS = 30.0


class IdleUnloader:
    """Periodically releases models unused for longer than the configured idle time."""

    def __init__(self, *, model_host: ModelHost, idle_seconds: float) -> None:
        """Store the model_host and the idle time; idle_seconds must be positive."""
        self._model_host = model_host
        self._idle_seconds = idle_seconds
        self._check_seconds = min(idle_seconds / 2, MAX_CHECK_SECONDS)

    async def run(self) -> None:
        """Check until cancelled; this loop is a task boundary that logs and survives failures."""
        while True:
            await asyncio.sleep(self._check_seconds)
            try:
                await self._model_host.unload_idle(idle_seconds=self._idle_seconds)
            except Exception:
                logger.exception("idle model unload failed")
