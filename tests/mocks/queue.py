"""Recording job queue, waiters and worker lock."""

import asyncio

from app.domain.ids import JobId
from app.domain.jobs import JobStatus


class MockJobQueue:
    """Records notifications and serves them from an asyncio queue."""

    def __init__(self) -> None:
        """Start empty."""
        self.notified: list[JobId] = []
        self._queue: asyncio.Queue[JobId] = asyncio.Queue()

    def notify(self, job_id: JobId) -> None:
        """Record and enqueue the job ID."""
        self.notified.append(job_id)
        self._queue.put_nowait(job_id)

    async def get(self) -> JobId:
        """Return the next job ID."""
        return await self._queue.get()

    def task_done(self) -> None:
        """Mark the last job processed."""
        self._queue.task_done()


class MockJobWaiters:
    """Records resolutions; waiting returns immediately."""

    def __init__(self) -> None:
        """Start empty."""
        self.resolved: list[tuple[JobId, JobStatus]] = []
        self.waits = 0

    async def wait(self, job_id: JobId, *, timeout_seconds: float) -> bool:
        """Count the wait and yield to the event loop."""
        self.waits += 1
        await asyncio.sleep(0)
        return False

    def resolve_if_terminal(self, job_id: JobId, status: JobStatus) -> None:
        """Record terminal resolutions."""
        if status.is_terminal:
            self.resolved.append((job_id, status))


class MockWorkerLock:
    """Lock whose availability is configurable."""

    def __init__(self, *, available: bool = True) -> None:
        """Configure whether acquire succeeds."""
        self.available = available
        self.held = False

    async def acquire(self) -> bool:
        """Take the lock if available."""
        self.held = self.available
        return self.held

    async def release(self) -> None:
        """Release the lock."""
        self.held = False
