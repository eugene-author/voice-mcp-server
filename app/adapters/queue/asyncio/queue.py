"""In-process wake-up queue and inline-wait futures."""

import asyncio
import logging
from collections import defaultdict

from app.domain.ids import JobId
from app.domain.jobs import JobStatus

logger = logging.getLogger(__name__)


class AsyncioJobQueue:
    """Bounded, de-duplicating asyncio queue of job IDs; a full queue drops the hint and polling recovers it."""

    def __init__(self, *, maxsize: int) -> None:
        """Create the bounded queue."""
        self._queue: asyncio.Queue[JobId] = asyncio.Queue(maxsize=maxsize)
        self._pending: set[JobId] = set()

    def notify(self, job_id: JobId) -> None:
        """Wake the worker for a committed job without blocking; IDs already waiting are not added twice."""
        if job_id in self._pending:
            return
        try:
            self._queue.put_nowait(job_id)
        except asyncio.QueueFull:
            logger.warning("job queue full; polling will pick the job up", extra={"job_id": str(job_id)})
            return
        self._pending.add(job_id)

    async def get(self) -> JobId:
        """Wait for the next woken job ID."""
        job_id = await self._queue.get()
        self._pending.discard(job_id)
        return job_id

    def task_done(self) -> None:
        """Mark the last job ID returned by get as processed."""
        self._queue.task_done()


class AsyncioJobWaiters:
    """Futures per job ID that tool calls await for a bounded time."""

    def __init__(self) -> None:
        """Start with no waiters."""
        self._futures: defaultdict[JobId, list[asyncio.Future[None]]] = defaultdict(list)

    async def wait(self, job_id: JobId, *, timeout_seconds: float) -> bool:
        """Wait until the job is resolved or the timeout passes; return whether it was resolved."""
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._futures[job_id].append(future)
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=timeout_seconds)
        except TimeoutError:
            return False
        finally:
            self._discard(job_id, future)
        return True

    def resolve_if_terminal(self, job_id: JobId, status: JobStatus) -> None:
        """Release waiters of a job that reached a terminal status."""
        if not status.is_terminal:
            return
        for future in self._futures.pop(job_id, []):
            if not future.done():
                future.set_result(None)

    def _discard(self, job_id: JobId, future: asyncio.Future[None]) -> None:
        """Forget one waiter's future."""
        futures = self._futures.get(job_id)
        if futures is None:
            return
        if future in futures:
            futures.remove(future)
        if not futures:
            self._futures.pop(job_id, None)
