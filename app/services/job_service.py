"""Command side of jobs: validate, persist queued jobs, wake the worker and wait briefly."""

import asyncio
import logging
from dataclasses import dataclass

from app.domain.caller import Caller
from app.domain.errors import DuplicateRequestError
from app.domain.ids import JobId, VoiceId
from app.domain.jobs import JobCommand, JobEvent, JobStatus, JobType, JobView
from app.ports.clock import Clock, MonotonicClock
from app.ports.job_queue import JobQueue, JobWaiters
from app.ports.runtime import RuntimeCatalog
from app.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from app.services.capabilities import HostCapabilities
from app.services.input_transfer import InputUploader
from app.services.job_planning import JobPlan, JobPlanner
from app.services.job_query_service import JobQueryService

logger = logging.getLogger(__name__)
WAIT_SLICE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class Submission:
    """Result of submitting a command."""

    job_id: JobId
    voice_id: VoiceId | None = None
    deduplicated: bool = False


class JobService:
    """Validates and persists job submissions, then wakes the worker."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkFactory,
        planner: JobPlanner,
        queue: JobQueue,
        waiters: JobWaiters,
        queries: JobQueryService,
        runtime: RuntimeCatalog,
        clock: Clock,
        monotonic: MonotonicClock,
        submitted: asyncio.Event | None = None,
        uploader: InputUploader | None = None,
    ) -> None:
        """Store collaborators; submitted requests a wake-controller tick, uploader copies inputs (ingest mode)."""
        self._uow = uow
        self._planner = planner
        self._queue = queue
        self._waiters = waiters
        self._queries = queries
        self._runtime = runtime
        self._clock = clock
        self._monotonic = monotonic
        self._submitted = submitted
        self._uploader = uploader

    async def submit(self, command: JobCommand, *, owner: Caller) -> Submission:
        """Validate, commit queued jobs (deduplicated by request_id), and notify only after commit."""
        HostCapabilities.from_runtime(self._runtime).require_for(command)
        plan = await self._planner.plan(command, owner=owner)
        if self._uploader is not None:
            async with self._uow() as uow:
                existing = await self._existing(uow, command, owner)
            if existing is not None:
                return existing
            plan = await self._uploader.upload(plan)
        try:
            async with self._uow() as uow:
                existing = await self._existing(uow, command, owner)
                if existing is not None:
                    return existing
                await self._persist(uow, plan)
        except DuplicateRequestError:
            async with self._uow() as uow:
                existing = await self._existing(uow, command, owner)
            if existing is None:
                raise
            return existing
        for job in plan.jobs:
            if job.type is not JobType.TRANSCRIBE_BATCH:
                self._queue.notify(job.id)
        if self._submitted is not None:
            self._submitted.set()
        voice_id = plan.voice.voice_id if plan.voice else None
        logger.info("job submitted", extra={"job_id": str(plan.primary_job_id), "jobs": len(plan.jobs)})
        return Submission(job_id=plan.primary_job_id, voice_id=voice_id)

    async def wait_for(self, job_id: JobId, *, owner: Caller, timeout_seconds: float) -> JobView:
        """Return the job once terminal, or its current state after the timeout."""
        deadline = self._monotonic.monotonic() + timeout_seconds
        while True:
            view = await self._queries.get(job_id, owner=owner)
            remaining = deadline - self._monotonic.monotonic()
            if view.status.is_terminal or remaining <= 0:
                return view
            await self._waiters.wait(job_id, timeout_seconds=min(remaining, WAIT_SLICE_SECONDS))

    async def _existing(self, uow: UnitOfWork, command: JobCommand, owner: Caller) -> Submission | None:
        """Return the earlier submission with the same request_id, if any."""
        if command.request_id is None:
            return None
        job_id = await uow.jobs.find_by_request_id(owner, command.request_id)
        if job_id is None:
            return None
        view = await uow.jobs.get(job_id, owner)
        voice_id = VoiceId(view.voice_id) if view and view.voice_id else None
        return Submission(job_id=job_id, voice_id=voice_id, deduplicated=True)

    async def _persist(self, uow: UnitOfWork, plan: JobPlan) -> None:
        """Insert the voice and job rows of a plan with a submitted event each."""
        if plan.voice is not None:
            await uow.voices.add(plan.voice)
        for job in plan.jobs:
            await uow.jobs.add(job)
            await uow.events.add(
                JobEvent(job_id=job.id, at=job.created_at, event_type="submitted", message=str(JobStatus.QUEUED))
            )
