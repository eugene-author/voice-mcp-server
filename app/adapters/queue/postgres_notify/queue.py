"""Job wake-ups across processes over PostgreSQL NOTIFY; jobs stay in PostgreSQL, so polling recovers a missed one."""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg

from app.adapters.queue.asyncio.queue import AsyncioJobQueue
from app.domain.ids import JobId

logger = logging.getLogger(__name__)
CHANNEL = "voice_mcp_jobs"
RECONNECT_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 5.0, 10.0)
KEEPALIVE_SECONDS = 30.0
PROBE_TIMEOUT_SECONDS = 5.0
CONNECTION_ERRORS = (OSError, ConnectionError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError)

type Connect = Callable[[], Awaitable[asyncpg.Connection]]


def connector(dsn: str, *, timeout_seconds: float) -> Connect:
    """Return a factory for dedicated asyncpg connections to a plain postgresql:// DSN."""

    async def connect() -> asyncpg.Connection:
        """Open one connection."""
        return await asyncpg.connect(dsn, timeout=timeout_seconds)

    return connect


class PostgresAnnouncingJobQueue:
    """Api-role JobQueue: notify buffers the job ID and a sender task announces it with pg_notify."""

    def __init__(self, *, connect: Connect, maxsize: int) -> None:
        """Create the bounded buffer; nothing connects until run starts."""
        self._connect = connect
        self._buffer: asyncio.Queue[JobId] = asyncio.Queue(maxsize=maxsize)

    def notify(self, job_id: JobId) -> None:
        """Buffer a committed job's ID without blocking; a full buffer drops it and the worker's poll recovers it."""
        try:
            self._buffer.put_nowait(job_id)
        except asyncio.QueueFull:
            logger.warning("wake-up buffer full; the worker's poll will pick the job up", extra={"job_id": str(job_id)})

    async def get(self) -> JobId:
        """Refuse: the api role does not consume wake-ups."""
        raise RuntimeError("the api role announces jobs and does not consume them")

    def task_done(self) -> None:
        """Refuse: the api role does not consume wake-ups."""
        raise RuntimeError("the api role announces jobs and does not consume them")

    async def run(self) -> None:
        """Send buffered IDs until cancelled, reconnecting with backoff; this loop is a task boundary."""
        attempt = 0
        while True:
            connection: asyncpg.Connection | None = None
            try:
                connection = await self._connect()
                attempt = 0
                while True:
                    job_id = await self._buffer.get()
                    try:
                        await connection.execute("SELECT pg_notify($1, $2)", CHANNEL, str(job_id))
                    except CONNECTION_ERRORS:
                        self.notify(job_id)
                        raise
            except CONNECTION_ERRORS as exc:
                logger.warning("job announcer disconnected; reconnecting", extra={"error": type(exc).__name__})
            finally:
                if connection is not None:
                    await _close(connection)
            await asyncio.sleep(RECONNECT_BACKOFF_SECONDS[min(attempt, len(RECONNECT_BACKOFF_SECONDS) - 1)])
            attempt += 1


class PostgresListeningJobQueue:
    """Worker-role JobQueue: LISTENs on a dedicated connection and feeds IDs into a de-duplicating local queue."""

    def __init__(
        self, *, connect: Connect, local: AsyncioJobQueue, keepalive_seconds: float = KEEPALIVE_SECONDS
    ) -> None:
        """Wrap the local queue; nothing connects until run starts."""
        self._connect = connect
        self._local = local
        self._keepalive = keepalive_seconds
        self.listening = asyncio.Event()
        """Set while a LISTEN connection is open."""

    def notify(self, job_id: JobId) -> None:
        """Wake the local worker, as recovery polling and the listener both do."""
        self._local.notify(job_id)

    async def get(self) -> JobId:
        """Wait for the next woken job ID."""
        return await self._local.get()

    def task_done(self) -> None:
        """Mark the last job ID returned by get as processed."""
        self._local.task_done()

    async def run(self) -> None:
        """Listen until cancelled, reconnecting with backoff after the connection drops; a task boundary."""
        attempt = 0
        while True:
            connection: asyncpg.Connection | None = None
            try:
                connection = await self._connect()
                closed = asyncio.Event()
                connection.add_termination_listener(lambda _connection, event=closed: event.set())
                await connection.add_listener(CHANNEL, self._on_notification)
                self.listening.set()
                attempt = 0
                await self._watch(connection, closed)
            except CONNECTION_ERRORS as exc:
                logger.warning("job listener disconnected; reconnecting", extra={"error": type(exc).__name__})
            finally:
                self.listening.clear()
                if connection is not None:
                    await _close(connection)
            await asyncio.sleep(RECONNECT_BACKOFF_SECONDS[min(attempt, len(RECONNECT_BACKOFF_SECONDS) - 1)])
            attempt += 1

    async def _watch(self, connection: asyncpg.Connection, closed: asyncio.Event) -> None:
        """Return when the connection closes; probe it periodically to notice silently dropped links."""
        while not closed.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(closed.wait(), timeout=self._keepalive)
            if not closed.is_set():
                await connection.execute("SELECT 1", timeout=PROBE_TIMEOUT_SECONDS)

    def _on_notification(self, _connection: Any, _pid: int, _channel: str, payload: str) -> None:
        """Wake the local worker for one announced job ID; malformed payloads are ignored."""
        try:
            job_id = JobId(uuid.UUID(payload))
        except ValueError:
            logger.warning("ignoring a malformed job notification")
            return
        self._local.notify(job_id)


async def _close(connection: asyncpg.Connection) -> None:
    """Close a connection, aborting it when a clean close fails."""
    try:
        await asyncio.wait_for(connection.close(), timeout=PROBE_TIMEOUT_SECONDS)
    except Exception:
        connection.terminate()
