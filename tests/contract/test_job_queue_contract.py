"""JobQueue and JobWaiters contracts."""

import asyncio

from app.adapters.queue.asyncio.queue import AsyncioJobQueue, AsyncioJobWaiters
from app.domain.ids import JobId, uuid7
from app.domain.jobs import JobStatus


async def test_queue_delivers_and_deduplicates_pending_ids() -> None:
    """Deliver notified IDs in order without duplicates while pending."""
    queue = AsyncioJobQueue(maxsize=10)
    first, second = JobId(uuid7()), JobId(uuid7())
    queue.notify(first)
    queue.notify(first)
    queue.notify(second)
    assert [await queue.get(), await queue.get()] == [first, second]
    queue.notify(first)
    assert await queue.get() == first


async def test_full_queue_drops_hints() -> None:
    """Drop hints when full; polling recovers them."""
    queue = AsyncioJobQueue(maxsize=1)
    queue.notify(JobId(uuid7()))
    queue.notify(JobId(uuid7()))
    await queue.get()
    assert queue._queue.empty()


async def test_waiters_resolve_on_terminal_status_only() -> None:
    """Release waiters for terminal statuses and time out otherwise."""
    waiters = AsyncioJobWaiters()
    job_id = JobId(uuid7())
    waiting = asyncio.create_task(waiters.wait(job_id, timeout_seconds=5))
    await asyncio.sleep(0)
    waiters.resolve_if_terminal(job_id, JobStatus.RUNNING)
    await asyncio.sleep(0)
    assert not waiting.done()
    waiters.resolve_if_terminal(job_id, JobStatus.COMPLETED)
    assert await waiting is True
    assert await waiters.wait(job_id, timeout_seconds=0.01) is False
