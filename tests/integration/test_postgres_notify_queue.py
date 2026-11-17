"""Cross-process wake-ups over PostgreSQL LISTEN/NOTIFY, including reconnects after dropped connections."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
import pytest

from app.adapters.queue.asyncio.queue import AsyncioJobQueue
from app.adapters.queue.postgres_notify.queue import (
    CHANNEL,
    PostgresAnnouncingJobQueue,
    PostgresListeningJobQueue,
    connector,
)
from app.db.engine import plain_database_url
from app.domain.ids import JobId, uuid7
from tests.waiting import eventually

pytestmark = pytest.mark.integration
TIMEOUT = 10.0


@asynccontextmanager
async def _running(*loops: object) -> AsyncIterator[None]:
    """Run the queues' loops in the background and cancel them afterwards."""
    tasks = [asyncio.create_task(loop.run()) for loop in loops]  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _terminate_listeners(dsn: str) -> int:
    """Kill every backend listening on the job channel, as a dropped network link would."""
    connection = await asyncpg.connect(dsn)
    try:
        rows = await connection.fetch(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query LIKE $1 AND pid <> pg_backend_pid()",
            f"%LISTEN%{CHANNEL}%",
        )
    finally:
        await connection.close()
    return len(rows)


async def test_announced_job_reaches_the_listening_worker(migrated_database: str) -> None:
    """Deliver an ID notified in the api process to the worker's local queue."""
    connect = connector(plain_database_url(migrated_database), timeout_seconds=5)
    announcer = PostgresAnnouncingJobQueue(connect=connect, maxsize=10)
    listener = PostgresListeningJobQueue(connect=connect, local=AsyncioJobQueue(maxsize=10))
    job_id = JobId(uuid7())
    async with _running(announcer, listener):
        await asyncio.wait_for(listener.listening.wait(), TIMEOUT)
        announcer.notify(job_id)
        assert await asyncio.wait_for(listener.get(), TIMEOUT) == job_id
        listener.task_done()


async def test_listener_reconnects_after_its_connection_drops(migrated_database: str) -> None:
    """Resume receiving after the LISTEN connection is killed, with no restart of the worker."""
    dsn = plain_database_url(migrated_database)
    connect = connector(dsn, timeout_seconds=5)
    announcer = PostgresAnnouncingJobQueue(connect=connect, maxsize=10)
    listener = PostgresListeningJobQueue(connect=connect, local=AsyncioJobQueue(maxsize=10))
    async with _running(announcer, listener):
        await asyncio.wait_for(listener.listening.wait(), TIMEOUT)
        assert await _terminate_listeners(dsn) == 1
        await eventually(lambda: not listener.listening.is_set(), within=TIMEOUT)
        await asyncio.wait_for(listener.listening.wait(), TIMEOUT)
        job_id = JobId(uuid7())
        announcer.notify(job_id)
        assert await asyncio.wait_for(listener.get(), TIMEOUT) == job_id


async def test_malformed_payloads_are_ignored(migrated_database: str) -> None:
    """Skip notifications that are not job IDs and keep listening."""
    dsn = plain_database_url(migrated_database)
    listener = PostgresListeningJobQueue(connect=connector(dsn, timeout_seconds=5), local=AsyncioJobQueue(maxsize=10))
    job_id = JobId(uuid7())
    async with _running(listener):
        await asyncio.wait_for(listener.listening.wait(), TIMEOUT)
        sender = await asyncpg.connect(dsn)
        try:
            await sender.execute("SELECT pg_notify($1, 'not-a-uuid')", CHANNEL)
            await sender.execute("SELECT pg_notify($1, $2)", CHANNEL, str(job_id))
        finally:
            await sender.close()
        assert await asyncio.wait_for(listener.get(), TIMEOUT) == job_id


async def test_announcer_does_not_consume() -> None:
    """Refuse get and task_done in the api role."""
    announcer = PostgresAnnouncingJobQueue(connect=connector("postgresql://unused", timeout_seconds=1), maxsize=1)
    with pytest.raises(RuntimeError, match="does not consume"):
        await announcer.get()
    with pytest.raises(RuntimeError, match="does not consume"):
        announcer.task_done()
