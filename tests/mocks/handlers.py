"""Job handler doubles that hold a job running until the test lets it finish."""

import asyncio
from collections.abc import Awaitable, Callable

from app.domain.jobs import ClaimedJob, JobCompletion, ProgressCallback
from app.workers.job_worker import JobHandler


class MockBlockingHandler:
    """Signals when a job starts, waits until released, then delegates to the real handler."""

    def __init__(self, inner: JobHandler) -> None:
        """Wrap the real handler."""
        self._inner = inner
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.cancelled = False

    async def execute(self, job: ClaimedJob, progress: ProgressCallback) -> JobCompletion:
        """Wait for finish, then run the real handler; record a cancellation."""
        self.started.set()
        try:
            await self.finish.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return await self._inner.execute(job, progress)


class MockScriptedHandler:
    """Runs a test's step while the job executes, then returns the completion or raises the error."""

    def __init__(
        self,
        *,
        during: Callable[[ClaimedJob], Awaitable[None]],
        completion: JobCompletion | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store the step and the result; without an error the completion, or an empty one, is returned."""
        self._during = during
        self._completion = completion or JobCompletion()
        self._error = error

    async def execute(self, job: ClaimedJob, progress: ProgressCallback) -> JobCompletion:
        """Run the step, then finish as configured without reporting progress."""
        await self._during(job)
        if self._error is not None:
            raise self._error
        return self._completion
