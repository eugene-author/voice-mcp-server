"""Job transitions in short units of work, and the register_voice job/voice rules."""

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta

from app.domain.errors import ClaimLostError, NotFoundError, VoiceMcpError
from app.domain.ids import JobId
from app.domain.jobs import ClaimedJob, JobCompletion, JobError, JobEvent, JobPhase, JobStatus, ProgressCallback
from app.ports.clock import Clock
from app.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory

logger = logging.getLogger(__name__)


class _VoiceNoLongerPending(Exception):
    """Raised inside a completion transaction to roll it back when the registered voice was deleted meanwhile."""


async def apply_registration_rules(
    uow: UnitOfWork, job_ids: Sequence[JobId], status: JobStatus, *, now: datetime
) -> None:
    """Move the voices of registration jobs that just reached status (failed, cancelled or queued by manual retry)."""
    voice_by_job = await uow.jobs.registration_voices(list(job_ids))
    for voice_id in voice_by_job.values():
        if status in (JobStatus.FAILED, JobStatus.CANCELLED):
            await uow.voices.mark_failed(voice_id, now=now)
        elif status is JobStatus.QUEUED:
            await uow.voices.reset_to_pending(voice_id, now=now)


class JobTransitions:
    """Persists job state transitions; does not hold a transaction across inference or uploads."""

    def __init__(
        self, *, uow: UnitOfWorkFactory, clock: Clock, retry_backoff_seconds: tuple[float, ...], save_events: bool
    ) -> None:
        """Store injected collaborators and retry policy."""
        self._uow = uow
        self._clock = clock
        self._backoff = retry_backoff_seconds or (5.0,)
        self._save_events = save_events

    async def claim(self, job_id: JobId, *, lease_owner: str) -> ClaimedJob | None:
        """Atomically claim a queued job; None when it is not claimable."""
        async with self._uow() as uow:
            now = self._clock.now()
            job = await uow.jobs.claim(job_id, lease_owner=lease_owner, now=now)
            if job is not None:
                await self._event(
                    uow, JobEvent(job_id=job_id, at=now, event_type="started", metadata={"attempt": job.attempt_count})
                )
        return job

    def progress_for(self, job: ClaimedJob) -> ProgressCallback:
        """Return a callback that records the job's phase and progress, and stops a claim that lost the job."""
        recorded_phase: JobPhase | None = None

        async def report(phase: JobPhase, progress: int) -> None:
            """Persist one update and log each new phase once, or raise ClaimLostError for a lost claim."""
            nonlocal recorded_phase
            async with self._uow() as uow:
                now = self._clock.now()
                held = await uow.jobs.set_phase(
                    job.id, claim_token=job.claim_token, phase=phase, progress=progress, now=now
                )
                if held and phase is not recorded_phase:
                    await self._event(uow, JobEvent(job_id=job.id, at=now, event_type="phase", phase=phase))
            if held:
                recorded_phase = phase
            if not held:
                raise ClaimLostError(f"claim of job {job.id} was lost")

        return report

    async def mark_completed(self, job: ClaimedJob, completion: JobCompletion) -> JobStatus | None:
        """Record assets, transcript, metrics and voice readiness in one transaction; None if the claim was lost."""
        try:
            return await self._complete(job, completion)
        except _VoiceNoLongerPending:
            logger.warning("voice was deleted before its registration finished", extra={"job_id": str(job.id)})
            return await self.mark_failed(job, NotFoundError("the voice was deleted before its registration finished"))

    async def mark_failed(self, job: ClaimedJob, error: VoiceMcpError) -> JobStatus | None:
        """Fail the job and, for registrations, its pending voice; None if the claim was lost."""
        async with self._uow() as uow:
            now = self._clock.now()
            if not await uow.jobs.fail(
                job.id, JobError(error.code, error.message), claim_token=job.claim_token, now=now
            ):
                logger.warning("claim was lost before failure", extra={"job_id": str(job.id)})
                return None
            await apply_registration_rules(uow, [job.id], JobStatus.FAILED, now=now)
            await self._event(uow, JobEvent(job_id=job.id, at=now, event_type="failed", message=error.code))
        return JobStatus.FAILED

    async def schedule_retry(self, job: ClaimedJob, error: VoiceMcpError) -> JobStatus | None:
        """Move the job to retry_wait, keeping a registration's voice pending; None if the claim was lost."""
        delay = self._backoff[min(job.attempt_count, len(self._backoff)) - 1]
        async with self._uow() as uow:
            now = self._clock.now()
            scheduled = await uow.jobs.schedule_retry(
                job.id,
                JobError(error.code, error.message),
                claim_token=job.claim_token,
                next_attempt_at=now + timedelta(seconds=delay),
            )
            if not scheduled:
                logger.warning("claim was lost before the retry", extra={"job_id": str(job.id)})
                return None
            await self._event(
                uow,
                JobEvent(
                    job_id=job.id, at=now, event_type="retry_scheduled", message=error.code, metadata={"delay": delay}
                ),
            )
        return JobStatus.RETRY_WAIT

    async def retry(self, job_id: JobId) -> bool:
        """Manually requeue a failed job with one more attempt; refused for the registration of a deleted voice."""
        async with self._uow() as uow:
            now = self._clock.now()
            for voice_id in (await uow.jobs.registration_voices([job_id])).values():
                if await uow.voices.get(voice_id, None) is None:
                    return False
            if not await uow.jobs.set_status(job_id, JobStatus.QUEUED, now=now):
                return False
            await apply_registration_rules(uow, [job_id], JobStatus.QUEUED, now=now)
            await self._event(uow, JobEvent(job_id=job_id, at=now, event_type="retried", message="manual"))
        return True

    async def release(self, job: ClaimedJob) -> bool:
        """Return a job interrupted by a drain deadline to queued without using up its attempt."""
        async with self._uow() as uow:
            now = self._clock.now()
            if not await uow.jobs.release(job.id, claim_token=job.claim_token):
                logger.warning("claim was lost before release", extra={"job_id": str(job.id)})
                return False
            await self._event(uow, JobEvent(job_id=job.id, at=now, event_type="released", message="drain_deadline"))
        return True

    async def cancel(self, job_id: JobId) -> bool:
        """Cancel a queued or retry-waiting job; a cancelled registration fails its pending voice."""
        async with self._uow() as uow:
            now = self._clock.now()
            if not await uow.jobs.set_status(job_id, JobStatus.CANCELLED, now=now):
                return False
            await apply_registration_rules(uow, [job_id], JobStatus.CANCELLED, now=now)
            await self._event(uow, JobEvent(job_id=job_id, at=now, event_type="cancelled", message="manual"))
        return True

    async def _complete(self, job: ClaimedJob, completion: JobCompletion) -> JobStatus | None:
        """Complete the job first so nothing is recorded for a claim that lost the job."""
        async with self._uow() as uow:
            now = self._clock.now()
            if not await uow.jobs.complete(job.id, completion, claim_token=job.claim_token, now=now):
                logger.warning("claim was lost before completion", extra={"job_id": str(job.id)})
                return None
            if completion.voice_version_id is not None:
                reference = completion.voice_reference
                ready = await uow.voices.mark_ready(
                    completion.voice_version_id,
                    reference=reference.stored.location,
                    sha256=reference.stored.sha256,
                    now=now,
                )
                if not ready:
                    raise _VoiceNoLongerPending
            for asset in completion.assets:
                await uow.assets.add(
                    asset,
                    owner=job.owner,
                    job_id=job.id,
                    voice_version_id=completion.voice_version_id,
                    created_at=now,
                )
            if completion.transcript is not None:
                await uow.transcripts.add(job.id, job.owner, completion.transcript, created_at=now)
            if completion.metrics is not None:
                await uow.metrics.add(job.id, completion.metrics, created_at=now)
            await self._event(
                uow, JobEvent(job_id=job.id, at=now, event_type="completed", message=str(completion.status))
            )
        return completion.status

    async def _event(self, uow: UnitOfWork, event: JobEvent) -> None:
        """Append a job event when event history is enabled."""
        if self._save_events:
            await uow.events.add(event)
