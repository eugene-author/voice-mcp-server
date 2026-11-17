"""Single-holder locks (worker, wake controller) backed by PostgreSQL session-level advisory locks."""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

logger = logging.getLogger(__name__)
WORKER_LOCK_KEY = 0x766F6963654D4350
"""Advisory lock key ('voiceMCP' as a 64-bit integer)."""

CONTROLLER_LOCK_KEY = 0x766F696365576B65
"""Advisory lock key of the wake controller ('voiceWke'), separate from the worker lock."""


class PostgresWorkerLock:
    """Holds an advisory lock on a dedicated connection for the process lifetime."""

    def __init__(self, engine: AsyncEngine, *, key: int = WORKER_LOCK_KEY) -> None:
        """Store the engine and lock key."""
        self._engine = engine
        self._key = key
        self._connection: AsyncConnection | None = None

    async def acquire(self) -> bool:
        """Try to take the lock without waiting; return whether it is held."""
        if self._connection is not None:
            return True
        connection = await self._engine.connect()
        try:
            acquired = bool(await connection.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": self._key}))
            await connection.commit()
        except BaseException:
            await connection.close()
            raise
        if not acquired:
            await connection.close()
            return False
        self._connection = connection
        return True

    async def release(self) -> None:
        """Release the lock and close its connection."""
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            await connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": self._key})
            await connection.commit()
        finally:
            await connection.close()
