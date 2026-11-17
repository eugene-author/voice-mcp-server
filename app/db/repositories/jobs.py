"""SQLAlchemy job repository with atomic claims; worker writes must match the claim token."""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import JobRow, MediaAssetRow
from app.domain.assets import PUBLIC_ROLES
from app.domain.caller import Caller
from app.domain.errors import DuplicateRequestError
from app.domain.ids import ClaimToken, JobId, VoiceId, new_claim_token
from app.domain.jobs import (
    BatchChild,
    BatchCounts,
    ClaimedJob,
    JobCompletion,
    JobCursor,
    JobError,
    JobFilter,
    JobPage,
    JobPhase,
    JobStatus,
    JobType,
    JobView,
    NewJob,
)
from app.domain.stt import Substitution
from app.domain.workers import JobDemand

MAX_FILES_PER_JOB = 20
FINISHED_STATUSES = (JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_ERRORS)


def sources_for(target: JobStatus) -> list[str]:
    """Return every status the state machine allows moving to target from."""
    return [status.value for status in JobStatus if status.can_transition_to(target)]


def _owned(owner: Caller) -> Any:
    """Return the ownership predicate for a caller."""
    return and_(JobRow.owner_issuer == owner.issuer, JobRow.owner_subject == owner.subject)


def _substitutions(rows: list[dict[str, Any]]) -> tuple[Substitution, ...]:
    """Rebuild substitutions from JSON."""
    return tuple(Substitution(**row) for row in rows)


def _claimed(row: JobRow, claim_token: ClaimToken) -> ClaimedJob:
    """Map a just-claimed job row to a ClaimedJob."""
    return ClaimedJob(
        id=JobId(row.id),
        type=JobType(row.type),
        owner=Caller(row.owner_issuer, row.owner_subject, row.owner_client_id),
        input_payload=row.input_payload,
        config_snapshot=row.config_snapshot,
        attempt_count=row.attempt_count,
        max_attempts=row.max_attempts,
        claim_token=claim_token,
        parent_job_id=JobId(row.parent_job_id) if row.parent_job_id else None,
        substitutions=_substitutions(row.substitutions or []),
    )


def _view(row: JobRow, files: dict[str, str]) -> JobView:
    """Map a job row and its public files to a JobView."""
    counts = row.counts
    return JobView(
        id=JobId(row.id),
        type=JobType(row.type),
        status=JobStatus(row.status),
        created_at=row.created_at,
        progress=row.progress,
        phase=JobPhase(row.phase) if row.phase else None,
        files=files,
        result=row.result_payload or {},
        warnings=tuple(row.warnings or ()),
        error=JobError(row.error_code, row.error_message or "") if row.error_code else None,
        parent_job_id=JobId(row.parent_job_id) if row.parent_job_id else None,
        counts=BatchCounts(**counts) if counts else None,
        voice_id=row.input_payload.get("voice_id") if row.type == JobType.REGISTER_VOICE.value else None,
    )


class SqlJobRepository:
    """JobRepository over the jobs table."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to a session owned by the unit of work."""
        self._session = session

    async def add(self, job: NewJob) -> None:
        """Insert a queued job."""
        self._session.add(
            JobRow(
                id=job.id,
                type=job.type.value,
                status=JobStatus.QUEUED.value,
                parent_job_id=job.parent_job_id,
                request_id=job.request_id,
                owner_issuer=job.owner.issuer,
                owner_subject=job.owner.subject,
                owner_client_id=job.owner.client_id,
                input_payload=job.input_payload,
                config_snapshot=job.config_snapshot,
                substitutions=[item.to_dict() for item in job.substitutions],
                warnings=[item.describe() for item in job.substitutions],
                max_attempts=job.max_attempts,
                attempt_count=0,
                progress=0,
                priority=0,
                created_at=job.created_at,
            )
        )
        try:
            await self._session.flush()
        except IntegrityError as exc:
            if "uq_jobs_owner_request_id" in str(exc.orig):
                raise DuplicateRequestError("request_id was already used") from exc
            raise

    async def find_by_request_id(self, owner: Caller, request_id: str) -> JobId | None:
        """Return the job previously submitted by this owner with this request_id."""
        job_id = await self._session.scalar(select(JobRow.id).where(_owned(owner), JobRow.request_id == request_id))
        return JobId(job_id) if job_id else None

    async def get(self, job_id: JobId, owner: Caller | None) -> JobView | None:
        """Return the job if it exists and is owned by owner; owner None is for trusted CLI use only."""
        query = select(JobRow).where(JobRow.id == job_id)
        if owner is not None:
            query = query.where(_owned(owner))
        row = await self._session.scalar(query)
        if row is None:
            return None
        files = await self._files([row])
        return _view(row, files.get(JobId(row.id), {}))

    async def page(self, owner: Caller, job_filter: JobFilter, *, limit: int, cursor: JobCursor | None) -> JobPage:
        """Return one keyset-paginated page of the owner's jobs, newest first."""
        query = select(JobRow).where(_owned(owner))
        if job_filter.status is not None:
            query = query.where(JobRow.status == job_filter.status.value)
        if job_filter.type is not None:
            query = query.where(JobRow.type == job_filter.type.value)
        if job_filter.parent_job_id is not None:
            query = query.where(JobRow.parent_job_id == job_filter.parent_job_id)
        if cursor is not None:
            query = query.where(
                or_(
                    JobRow.created_at < cursor.created_at,
                    and_(JobRow.created_at == cursor.created_at, JobRow.id < cursor.job_id),
                )
            )
        rows = list(
            await self._session.scalars(query.order_by(JobRow.created_at.desc(), JobRow.id.desc()).limit(limit + 1))
        )
        page, has_more = rows[:limit], len(rows) > limit
        files = await self._files(page)
        items = tuple(_view(row, files.get(JobId(row.id), {})) for row in page)
        next_cursor = JobCursor(page[-1].created_at, JobId(page[-1].id)).encode() if has_more and page else None
        return JobPage(items=items, next_cursor=next_cursor)

    async def claim(self, job_id: JobId, *, lease_owner: str, now: datetime) -> ClaimedJob | None:
        """Atomically move a queued, non-batch job to running under a new claim token."""
        claim_token = new_claim_token()
        row = await self._session.scalar(
            update(JobRow)
            .where(
                JobRow.id == job_id,
                JobRow.status == JobStatus.QUEUED.value,
                JobRow.type != JobType.TRANSCRIBE_BATCH.value,
            )
            .values(
                status=JobStatus.RUNNING.value,
                started_at=now,
                heartbeat_at=now,
                lease_owner=lease_owner,
                claim_token=claim_token,
                attempt_count=JobRow.attempt_count + 1,
                phase=None,
                progress=0,
                next_attempt_at=None,
            )
            .returning(JobRow)
        )
        return _claimed(row, claim_token) if row is not None else None

    async def set_phase(
        self, job_id: JobId, *, claim_token: ClaimToken, phase: JobPhase, progress: int, now: datetime
    ) -> bool:
        """Record the phase and progress of a job this claim holds; return whether it is still held."""
        return await self._update_held(
            job_id, claim_token, phase=phase.value, progress=max(0, min(progress, 100)), heartbeat_at=now
        )

    async def heartbeat(self, job_id: JobId, *, claim_token: ClaimToken, now: datetime) -> bool:
        """Refresh heartbeat_at of a job this claim holds; return whether it is still held."""
        return await self._update_held(job_id, claim_token, heartbeat_at=now)

    async def complete(
        self, job_id: JobId, completion: JobCompletion, *, claim_token: ClaimToken, now: datetime
    ) -> bool:
        """Move a job this claim holds to the completion status with its result; return whether it transitioned."""
        execution = completion.execution
        existing = await self._session.scalar(select(JobRow.warnings).where(JobRow.id == job_id))
        warnings = list(dict.fromkeys([*(existing or []), *completion.warnings]))
        values: dict[str, Any] = {
            "result_payload": completion.result,
            "warnings": warnings,
            "finished_at": now,
            "progress": 100,
            "phase": None,
            "error_code": None,
            "error_message": None,
        }
        if execution is not None:
            values |= {
                "backend": execution.backend,
                "model_id": execution.model_id,
                "model_revision": execution.model_revision,
                "device": execution.device,
                "compute_type": execution.compute_type,
            }
        return await self._update_held(job_id, claim_token, status=completion.status.value, **values)

    async def fail(self, job_id: JobId, error: JobError, *, claim_token: ClaimToken, now: datetime) -> bool:
        """Move a job this claim holds to failed; return whether it transitioned."""
        return await self._update_held(
            job_id,
            claim_token,
            status=JobStatus.FAILED.value,
            error_code=error.code,
            error_message=error.message,
            finished_at=now,
            phase=None,
        )

    async def schedule_retry(
        self, job_id: JobId, error: JobError, *, claim_token: ClaimToken, next_attempt_at: datetime
    ) -> bool:
        """Move a job this claim holds to retry_wait; return whether it transitioned."""
        return await self._update_held(
            job_id,
            claim_token,
            status=JobStatus.RETRY_WAIT.value,
            error_code=error.code,
            error_message=error.message,
            next_attempt_at=next_attempt_at,
        )

    async def requeue_due(self, *, now: datetime) -> int:
        """Move retry_wait jobs whose next attempt is due back to queued; return the count."""
        result = await self._session.execute(
            update(JobRow)
            .where(JobRow.status == JobStatus.RETRY_WAIT.value, JobRow.next_attempt_at <= now)
            .values(status=JobStatus.QUEUED.value, phase=None)
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def queued_ids(self, *, limit: int) -> list[JobId]:
        """Return queued, non-parent job IDs, oldest first."""
        rows = await self._session.scalars(
            select(JobRow.id)
            .where(JobRow.status == JobStatus.QUEUED.value, JobRow.type != JobType.TRANSCRIBE_BATCH.value)
            .order_by(JobRow.priority.desc(), JobRow.created_at, JobRow.id)
            .limit(limit)
        )
        return [JobId(job_id) for job_id in rows]

    async def interrupt_stale(self, *, heartbeat_before: datetime) -> list[JobId]:
        """Mark running jobs with an expired heartbeat as interrupted; return their IDs."""
        rows = await self._session.scalars(
            update(JobRow)
            .where(JobRow.status == JobStatus.RUNNING.value, JobRow.heartbeat_at < heartbeat_before)
            .values(status=JobStatus.INTERRUPTED.value, phase=None, lease_owner=None)
            .returning(JobRow.id)
        )
        return [JobId(job_id) for job_id in rows]

    async def resolve_interrupted(self, *, now: datetime) -> tuple[list[JobId], list[JobId]]:
        """Requeue interrupted jobs with attempts left and fail the rest; return (requeued, failed)."""
        requeued = await self._session.scalars(
            update(JobRow)
            .where(JobRow.status == JobStatus.INTERRUPTED.value, JobRow.attempt_count < JobRow.max_attempts)
            .values(status=JobStatus.QUEUED.value)
            .returning(JobRow.id)
        )
        requeued_ids = [JobId(job_id) for job_id in requeued]
        failed = await self._session.scalars(
            update(JobRow)
            .where(JobRow.status == JobStatus.INTERRUPTED.value)
            .values(
                status=JobStatus.FAILED.value,
                error_code="interrupted",
                error_message="the job was interrupted and has no attempts left",
                finished_at=now,
            )
            .returning(JobRow.id)
        )
        return requeued_ids, [JobId(job_id) for job_id in failed]

    async def batch_counts(self, parent_id: JobId) -> BatchCounts:
        """Return the child status counts of a batch parent."""
        rows = await self._session.execute(
            select(JobRow.status, func.count()).where(JobRow.parent_job_id == parent_id).group_by(JobRow.status)
        )
        count_by_status = {JobStatus(status): int(count) for status, count in rows}
        completed = sum(count_by_status.get(status, 0) for status in FINISHED_STATUSES)
        failed = count_by_status.get(JobStatus.FAILED, 0) + count_by_status.get(JobStatus.CANCELLED, 0)
        total = sum(count_by_status.values())
        return BatchCounts(total=total, completed=completed, failed=failed, queued=total - completed - failed)

    async def update_batch_parent(
        self, parent_id: JobId, *, counts: BatchCounts, status: JobStatus | None, now: datetime
    ) -> bool:
        """Store aggregated counts and, if given, the parent's new status; return whether the status changed."""
        progress = round(100 * counts.finished / counts.total) if counts.total else 100
        await self._session.execute(
            update(JobRow).where(JobRow.id == parent_id).values(counts=counts.to_dict(), progress=progress)
        )
        if status is None:
            return False
        values: dict[str, Any] = {"finished_at": now} if status.is_terminal else {"started_at": now}
        return await self._transition(parent_id, status, values)

    async def batch_children(self, parent_id: JobId) -> list[BatchChild]:
        """Return every child of a batch parent in creation order."""
        rows = list(
            await self._session.scalars(
                select(JobRow).where(JobRow.parent_job_id == parent_id).order_by(JobRow.created_at, JobRow.id)
            )
        )
        files = await self._files(rows)
        return [
            BatchChild(
                job_id=JobId(row.id),
                input_path=str(row.input_payload.get("input_path", "")),
                status=JobStatus(row.status),
                files=files.get(JobId(row.id), {}),
                error=JobError(row.error_code, row.error_message or "") if row.error_code else None,
            )
            for row in rows
        ]

    async def open_batch_parent_ids(self) -> list[JobId]:
        """Return batch parents that are not yet terminal."""
        rows = await self._session.scalars(
            select(JobRow.id).where(
                JobRow.type == JobType.TRANSCRIBE_BATCH.value,
                JobRow.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
            )
        )
        return [JobId(job_id) for job_id in rows]

    async def set_status(self, job_id: JobId, status: JobStatus, *, now: datetime) -> bool:
        """Apply an allowed manual transition such as a CLI cancel or retry; return whether it applied."""
        values: dict[str, Any] = {"phase": None}
        if status is JobStatus.CANCELLED:
            values["finished_at"] = now
        if status is JobStatus.QUEUED:
            values |= {
                "next_attempt_at": None,
                "finished_at": None,
                "error_code": None,
                "error_message": None,
                "max_attempts": JobRow.attempt_count + 1,
            }
        return await self._transition(job_id, status, values)

    async def registration_voices(self, job_ids: list[JobId]) -> dict[JobId, VoiceId]:
        """Return the voice of each register_voice job among job_ids; other job types are left out."""
        if not job_ids:
            return {}
        rows = await self._session.execute(
            select(JobRow.id, JobRow.input_payload["voice_id"].astext).where(
                JobRow.id.in_(job_ids), JobRow.type == JobType.REGISTER_VOICE.value
            )
        )
        return {JobId(job_id): VoiceId(voice_id) for job_id, voice_id in rows if voice_id}

    async def owner_of(self, job_id: JobId) -> Caller | None:
        """Return the owner of a job, for trusted internal use."""
        row = (
            await self._session.execute(
                select(JobRow.owner_issuer, JobRow.owner_subject, JobRow.owner_client_id).where(JobRow.id == job_id)
            )
        ).first()
        return Caller(row[0], row[1], row[2]) if row else None

    async def demand(self, *, now: datetime, heartbeat_before: datetime) -> JobDemand:
        """Count runnable jobs (queued, due retries, stale running) and running jobs with a fresh heartbeat."""
        stale = and_(JobRow.status == JobStatus.RUNNING.value, JobRow.heartbeat_at < heartbeat_before)
        runnable = or_(
            JobRow.status == JobStatus.QUEUED.value,
            and_(JobRow.status == JobStatus.RETRY_WAIT.value, JobRow.next_attempt_at <= now),
            stale,
        )
        fresh = and_(JobRow.status == JobStatus.RUNNING.value, JobRow.heartbeat_at >= heartbeat_before)
        row = (
            await self._session.execute(
                select(func.count().filter(runnable), func.count().filter(fresh)).where(
                    JobRow.type != JobType.TRANSCRIBE_BATCH.value
                )
            )
        ).one()
        return JobDemand(runnable=int(row[0]), running=int(row[1]))

    async def release(self, job_id: JobId, *, claim_token: ClaimToken) -> bool:
        """Return a job this claim holds to queued without using up its attempt; return whether it did."""
        return await self._update_held(
            job_id,
            claim_token,
            status=JobStatus.QUEUED.value,
            attempt_count=func.greatest(JobRow.attempt_count - 1, 0),
            lease_owner=None,
            phase=None,
            progress=0,
            heartbeat_at=None,
            started_at=None,
        )

    async def _update_held(self, job_id: JobId, claim_token: ClaimToken, **values: Any) -> bool:
        """Update the job only while this claim is the running one; return whether a row changed."""
        changed = await self._session.scalar(
            update(JobRow)
            .where(JobRow.id == job_id, JobRow.status == JobStatus.RUNNING.value, JobRow.claim_token == claim_token)
            .values(**values)
            .returning(JobRow.id)
        )
        return changed is not None

    async def _transition(self, job_id: JobId, target: JobStatus, values: dict[str, Any]) -> bool:
        """Update the job only if its current status may move to target; return whether a row changed."""
        changed = await self._session.scalar(
            update(JobRow)
            .where(JobRow.id == job_id, JobRow.status.in_(sources_for(target)))
            .values(status=target.value, **values)
            .returning(JobRow.id)
        )
        return changed is not None

    async def _files(self, rows: list[JobRow]) -> dict[JobId, dict[str, str]]:
        """Return public files of finished jobs, capped per job."""
        finished = [row.id for row in rows if JobStatus(row.status) in FINISHED_STATUSES]
        if not finished:
            return {}
        assets = await self._session.execute(
            select(MediaAssetRow.job_id, MediaAssetRow.name, MediaAssetRow.uri)
            .where(MediaAssetRow.job_id.in_(finished), MediaAssetRow.role.in_([role.value for role in PUBLIC_ROLES]))
            .order_by(MediaAssetRow.created_at, MediaAssetRow.name)
        )
        files: dict[JobId, dict[str, str]] = {}
        for job_uuid, name, uri in assets:
            if job_uuid is None:
                continue
            job_files = files.setdefault(JobId(job_uuid), {})
            if len(job_files) < MAX_FILES_PER_JOB:
                job_files[name] = uri
        return files
