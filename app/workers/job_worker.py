"""Single worker that processes woken job IDs one at a time; a drain release or lost claim is not a job failure."""

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from typing import Protocol

from app.domain.errors import ClaimLostError, InternalError, VoiceMcpError
from app.domain.ids import JobId
from app.domain.jobs import ClaimedJob, JobCompletion, JobStatus, JobType, ProgressCallback
from app.ports.job_queue import JobQueue, JobWaiters
from app.services.batch_service import BatchService
from app.services.job_workspace import JobWorkspace
from app.workers.heartbeat import Heartbeat
from app.workers.job_transitions import JobTransitions

logger = logging.getLogger(__name__)


class JobHandler(Protocol):
    """Executes one claimed job of a specific type; the transcription, synthesis and registration services fit it."""

    async def execute(self, job: ClaimedJob, progress: ProgressCallback) -> JobCompletion:
        """Run the job and return its completion."""
        ...


class _Released:
    """Marker result of a handler cancelled to release its job at a drain deadline."""


class _ClaimLost:
    """Marker result of a handler stopped because its claim no longer holds the job."""


RELEASED = _Released()
CLAIM_LOST = _ClaimLost()


class JobWorker:
    """Claims, executes and finalizes jobs sequentially; claiming can be paused for a drain."""

    def __init__(
        self,
        *,
        transitions: JobTransitions,
        handlers: Mapping[JobType, JobHandler],
        queue: JobQueue,
        waiters: JobWaiters,
        batches: BatchService,
        workspace: JobWorkspace,
        heartbeat: Heartbeat,
        lease_owner: str,
    ) -> None:
        """Store injected collaborators and the handler registry."""
        self._heartbeat = heartbeat
        self._transitions = transitions
        self._handlers = handlers
        self._queue = queue
        self._waiters = waiters
        self._batches = batches
        self._workspace = workspace
        self._lease_owner = lease_owner
        self._claiming = asyncio.Event()
        self._claiming.set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._handler: asyncio.Task[JobCompletion] | None = None
        self.activity = 0
        """Incremented whenever a job is claimed or finishes, so callers can track the last activity."""

    @property
    def busy(self) -> bool:
        """Return whether a job is being claimed or executed."""
        return not self._idle.is_set()

    def stop_claiming(self) -> None:
        """Stop taking new jobs; woken IDs are dropped because PostgreSQL keeps those jobs queued."""
        self._claiming.clear()

    def resume_claiming(self) -> None:
        """Take new jobs again after a cancelled drain."""
        self._claiming.set()

    async def wait_idle(self) -> None:
        """Wait until no job is being processed."""
        await self._idle.wait()

    async def release_current(self) -> bool:
        """Cancel the running handler to release its job, wait until idle, and return whether one was cancelled."""
        cancelled = self._handler is not None and self._handler.cancel()
        await self._idle.wait()
        return cancelled

    async def run(self) -> None:
        """Process wake-ups until cancelled, skipping them while claiming is stopped."""
        while True:
            await self._claiming.wait()
            job_id = await self._queue.get()
            try:
                if self._claiming.is_set():
                    self._idle.clear()
                    await self._process(job_id)
            finally:
                self._idle.set()
                self._queue.task_done()

    async def _process(self, job_id: JobId) -> None:
        """Claim, execute and finalize one job; this is the task boundary that contains and logs failures."""
        try:
            job = await self._transitions.claim(job_id, lease_owner=self._lease_owner)
        except Exception:
            logger.exception("could not claim job", extra={"job_id": str(job_id)})
            return
        if job is None:
            return
        self.activity += 1
        try:
            result = await self._execute(job)
            if isinstance(result, _Released):
                await self._release(job)
                return
            status = None if isinstance(result, _ClaimLost) else await self._finalize(job, result)
        except Exception:
            logger.exception("could not record job completion; recovery will retry it", extra={"job_id": str(job_id)})
            return
        self.activity += 1
        if status is None:
            logger.warning("stopped a job whose claim was lost", extra={"job_id": str(job_id)})
            self._workspace.remove_claim(job)
            return
        self._waiters.resolve_if_terminal(job_id, status)
        if not status.is_terminal:
            return
        self._workspace.remove_job(job.id)
        if job.parent_job_id is None:
            return
        try:
            await self._batches.aggregate(job.parent_job_id)
        except Exception:
            logger.exception("batch aggregation failed; recovery will retry it", extra={"job_id": str(job_id)})

    async def _release(self, job: ClaimedJob) -> None:
        """Hand a job interrupted at the drain deadline back to the queue with its attempt unused."""
        logger.warning("released the running job at the drain deadline", extra={"job_id": str(job.id)})
        self.activity += 1
        self._workspace.remove_claim(job)
        await self._transitions.release(job)

    async def _execute(self, job: ClaimedJob) -> JobCompletion | VoiceMcpError | _Released | _ClaimLost:
        """Run the handler while keeping the lease; return the completion, the error, RELEASED or CLAIM_LOST."""
        handler = self._handlers.get(job.type)
        if handler is None:
            return InternalError(f"no handler registered for {job.type}")
        running = asyncio.create_task(
            handler.execute(job, self._transitions.progress_for(job)), name=f"handler-{job.id}"
        )
        self._handler = running
        claim_lost = False

        def stop_lost_claim() -> None:
            """Remember why the handler is cancelled, then cancel it."""
            nonlocal claim_lost
            claim_lost = True
            running.cancel()

        heartbeat = asyncio.create_task(
            self._heartbeat.keep_alive(job, on_lost=stop_lost_claim), name=f"heartbeat-{job.id}"
        )
        try:
            return await running
        except ClaimLostError:
            return CLAIM_LOST
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if running.cancelled() and current is not None and not current.cancelling():
                return CLAIM_LOST if claim_lost else RELEASED
            raise
        except VoiceMcpError as exc:
            logger.warning("job failed", extra={"job_id": str(job.id), "code": exc.code, "retryable": exc.retryable})
            return exc
        except Exception as exc:
            logger.exception("job failed unexpectedly", extra={"job_id": str(job.id)})
            return InternalError.from_exception(exc)
        finally:
            self._handler = None
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _finalize(self, job: ClaimedJob, result: JobCompletion | VoiceMcpError) -> JobStatus | None:
        """Record completion, a scheduled retry or a failure; None if the claim was lost."""
        if isinstance(result, JobCompletion):
            return await self._transitions.mark_completed(job, result)
        if result.retryable and job.has_attempts_left:
            return await self._transitions.schedule_retry(job, result)
        return await self._transitions.mark_failed(job, result)
