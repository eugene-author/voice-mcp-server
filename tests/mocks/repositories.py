"""In-memory repositories mirroring the SQL implementations' semantics, and their shared state."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from app.domain.assets import PUBLIC_ROLES, AssetLocation, PublishedAsset, UploadedInput
from app.domain.caller import Caller
from app.domain.errors import DuplicateRequestError
from app.domain.ids import ClaimToken, JobId, VoiceId, VoiceVersionId, new_claim_token
from app.domain.jobs import (
    BatchChild,
    BatchCounts,
    ClaimedJob,
    JobCompletion,
    JobCursor,
    JobError,
    JobEvent,
    JobFilter,
    JobPage,
    JobPhase,
    JobStatus,
    JobType,
    JobView,
    NewJob,
)
from app.domain.stt import TranscriptionResult
from app.domain.voices import NewVoice, Voice, VoiceStatus, VoiceVersion
from app.domain.workers import CapabilitySnapshot, JobDemand, WorkerRecord, WorkerStatus

FINISHED = (JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_ERRORS)


@dataclass
class JobRecord:
    """Mutable stored job."""

    job: NewJob
    status: JobStatus = JobStatus.QUEUED
    attempt_count: int = 0
    max_attempts: int = 2
    phase: JobPhase | None = None
    progress: int = 0
    lease_owner: str | None = None
    claim_token: ClaimToken | None = None
    heartbeat_at: datetime | None = None
    next_attempt_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: JobError | None = None
    counts: BatchCounts | None = None
    execution: Any = None


@dataclass
class AssetRecord:
    """Stored asset metadata."""

    asset: PublishedAsset
    owner: Caller
    job_id: JobId | None
    voice_version_id: VoiceVersionId | None


@dataclass
class Database:
    """All in-memory tables."""

    jobs: dict[JobId, JobRecord] = field(default_factory=dict)
    events: list[JobEvent] = field(default_factory=list)
    assets: list[AssetRecord] = field(default_factory=list)
    transcripts: dict[JobId, TranscriptionResult] = field(default_factory=dict)
    metrics: dict[JobId, dict[str, Any]] = field(default_factory=dict)
    voices: dict[VoiceId, Voice] = field(default_factory=dict)
    versions: dict[VoiceVersionId, VoiceVersion] = field(default_factory=dict)
    workers: dict[str, WorkerRecord] = field(default_factory=dict)


def _owned(record: JobRecord, owner: Caller | None) -> bool:
    """Return whether owner may see the record."""
    return owner is None or (record.job.owner.issuer, record.job.owner.subject) == (owner.issuer, owner.subject)


class MockJobRepository:
    """In-memory JobRepository."""

    def __init__(self, db: Database) -> None:
        """Bind to the shared tables."""
        self._db = db

    def _view(self, record: JobRecord) -> JobView:
        """Map a record to a JobView."""
        job = record.job
        files: dict[str, str] = {}
        if record.status in FINISHED:
            files = {
                a.asset.name: a.asset.stored.location.uri
                for a in self._db.assets
                if a.job_id == job.id and a.asset.role in PUBLIC_ROLES
            }
        return JobView(
            id=job.id,
            type=job.type,
            status=record.status,
            created_at=job.created_at,
            progress=record.progress,
            phase=record.phase,
            files=files,
            result=dict(record.result),
            warnings=tuple(record.warnings),
            error=record.error,
            parent_job_id=job.parent_job_id,
            counts=record.counts,
            voice_id=job.input_payload.get("voice_id") if job.type is JobType.REGISTER_VOICE else None,
        )

    def _held(self, job_id: JobId, claim_token: ClaimToken) -> JobRecord | None:
        """Return the job only while this claim is the running one."""
        record = self._db.jobs.get(job_id)
        held = record is not None and record.status is JobStatus.RUNNING and record.claim_token == claim_token
        return record if held else None

    def _transition(self, job_id: JobId, target: JobStatus) -> JobRecord | None:
        """Move the job to target if allowed and return it."""
        record = self._db.jobs.get(job_id)
        if record is None or not record.status.can_transition_to(target):
            return None
        record.status = target
        return record

    async def add(self, job: NewJob) -> None:
        """Insert a queued job, enforcing request_id uniqueness per owner."""
        if job.request_id is not None and await self.find_by_request_id(job.owner, job.request_id):
            raise DuplicateRequestError("request_id was already used")
        self._db.jobs[job.id] = JobRecord(
            job=job, max_attempts=job.max_attempts, warnings=[s.describe() for s in job.substitutions]
        )

    async def find_by_request_id(self, owner: Caller, request_id: str) -> JobId | None:
        """Return the owner's job with this request_id."""
        for record in self._db.jobs.values():
            if record.job.request_id == request_id and _owned(record, owner):
                return record.job.id
        return None

    async def get(self, job_id: JobId, owner: Caller | None) -> JobView | None:
        """Return the job if owned."""
        record = self._db.jobs.get(job_id)
        return self._view(record) if record is not None and _owned(record, owner) else None

    async def page(self, owner: Caller, job_filter: JobFilter, *, limit: int, cursor: JobCursor | None) -> JobPage:
        """Return one page, newest first."""
        records = sorted(
            (r for r in self._db.jobs.values() if _owned(r, owner)),
            key=lambda r: (r.job.created_at, r.job.id),
            reverse=True,
        )
        if job_filter.status is not None:
            records = [r for r in records if r.status is job_filter.status]
        if job_filter.type is not None:
            records = [r for r in records if r.job.type is job_filter.type]
        if job_filter.parent_job_id is not None:
            records = [r for r in records if r.job.parent_job_id == job_filter.parent_job_id]
        if cursor is not None:
            records = [r for r in records if (r.job.created_at, r.job.id) < (cursor.created_at, cursor.job_id)]
        page = records[:limit]
        next_cursor = (
            JobCursor(page[-1].job.created_at, page[-1].job.id).encode() if len(records) > limit and page else None
        )
        return JobPage(items=tuple(self._view(r) for r in page), next_cursor=next_cursor)

    async def claim(self, job_id: JobId, *, lease_owner: str, now: datetime) -> ClaimedJob | None:
        """Claim a queued non-batch job."""
        record = self._db.jobs.get(job_id)
        if record is None or record.status is not JobStatus.QUEUED or record.job.type is JobType.TRANSCRIBE_BATCH:
            return None
        record.status, record.lease_owner, record.heartbeat_at = JobStatus.RUNNING, lease_owner, now
        claim_token = record.claim_token = new_claim_token()
        record.attempt_count += 1
        job = record.job
        return ClaimedJob(
            id=job.id,
            type=job.type,
            owner=job.owner,
            input_payload=deepcopy(job.input_payload),
            config_snapshot=deepcopy(job.config_snapshot),
            attempt_count=record.attempt_count,
            max_attempts=record.max_attempts,
            claim_token=claim_token,
            parent_job_id=job.parent_job_id,
            substitutions=job.substitutions,
        )

    async def set_phase(
        self, job_id: JobId, *, claim_token: ClaimToken, phase: JobPhase, progress: int, now: datetime
    ) -> bool:
        """Record phase and progress of a job this claim holds."""
        record = self._held(job_id, claim_token)
        if record is None:
            return False
        record.phase, record.progress, record.heartbeat_at = phase, progress, now
        return True

    async def heartbeat(self, job_id: JobId, *, claim_token: ClaimToken, now: datetime) -> bool:
        """Refresh the heartbeat of a job this claim holds."""
        record = self._held(job_id, claim_token)
        if record is None:
            return False
        record.heartbeat_at = now
        return True

    async def complete(
        self, job_id: JobId, completion: JobCompletion, *, claim_token: ClaimToken, now: datetime
    ) -> bool:
        """Complete a job this claim holds."""
        record = self._held(job_id, claim_token)
        if record is None:
            return False
        record.status = completion.status
        record.result, record.finished_at, record.progress, record.phase = dict(completion.result), now, 100, None
        record.warnings = list(dict.fromkeys([*record.warnings, *completion.warnings]))
        record.error, record.execution = None, completion.execution
        return True

    async def fail(self, job_id: JobId, error: JobError, *, claim_token: ClaimToken, now: datetime) -> bool:
        """Fail a job this claim holds."""
        record = self._held(job_id, claim_token)
        if record is None:
            return False
        record.status, record.error, record.finished_at, record.phase = JobStatus.FAILED, error, now, None
        return True

    async def schedule_retry(
        self, job_id: JobId, error: JobError, *, claim_token: ClaimToken, next_attempt_at: datetime
    ) -> bool:
        """Move a job this claim holds to retry_wait."""
        record = self._held(job_id, claim_token)
        if record is None:
            return False
        record.status, record.error, record.next_attempt_at = JobStatus.RETRY_WAIT, error, next_attempt_at
        return True

    async def requeue_due(self, *, now: datetime) -> int:
        """Requeue due retries."""
        due = [
            r
            for r in self._db.jobs.values()
            if r.status is JobStatus.RETRY_WAIT and r.next_attempt_at and r.next_attempt_at <= now
        ]
        for record in due:
            record.status = JobStatus.QUEUED
        return len(due)

    async def queued_ids(self, *, limit: int) -> list[JobId]:
        """Return queued non-batch job IDs, oldest first."""
        queued = sorted(
            (
                r
                for r in self._db.jobs.values()
                if r.status is JobStatus.QUEUED and r.job.type is not JobType.TRANSCRIBE_BATCH
            ),
            key=lambda r: (r.job.created_at, r.job.id),
        )
        return [r.job.id for r in queued[:limit]]

    async def interrupt_stale(self, *, heartbeat_before: datetime) -> list[JobId]:
        """Interrupt running jobs whose heartbeat expired."""
        stale = [
            r
            for r in self._db.jobs.values()
            if r.status is JobStatus.RUNNING and r.heartbeat_at is not None and r.heartbeat_at < heartbeat_before
        ]
        for record in stale:
            record.status, record.lease_owner = JobStatus.INTERRUPTED, None
        return [r.job.id for r in stale]

    async def resolve_interrupted(self, *, now: datetime) -> tuple[list[JobId], list[JobId]]:
        """Requeue or fail interrupted jobs."""
        requeued: list[JobId] = []
        failed: list[JobId] = []
        for record in self._db.jobs.values():
            if record.status is not JobStatus.INTERRUPTED:
                continue
            if record.attempt_count < record.max_attempts:
                record.status = JobStatus.QUEUED
                requeued.append(record.job.id)
            else:
                record.status, record.finished_at = JobStatus.FAILED, now
                record.error = JobError("interrupted", "the job was interrupted and has no attempts left")
                failed.append(record.job.id)
        return requeued, failed

    async def batch_counts(self, parent_id: JobId) -> BatchCounts:
        """Return child status counts."""
        children = [r for r in self._db.jobs.values() if r.job.parent_job_id == parent_id]
        completed = sum(r.status in FINISHED for r in children)
        failed = sum(r.status in (JobStatus.FAILED, JobStatus.CANCELLED) for r in children)
        return BatchCounts(
            total=len(children), completed=completed, failed=failed, queued=len(children) - completed - failed
        )

    async def update_batch_parent(
        self, parent_id: JobId, *, counts: BatchCounts, status: JobStatus | None, now: datetime
    ) -> bool:
        """Store counts and optionally transition the parent."""
        record = self._db.jobs[parent_id]
        record.counts = counts
        if status is None:
            return False
        changed = self._transition(parent_id, status) is not None
        if changed and status.is_terminal:
            record.finished_at = now
        return changed

    async def batch_children(self, parent_id: JobId) -> list[BatchChild]:
        """Return children in creation order."""
        children = sorted(
            (r for r in self._db.jobs.values() if r.job.parent_job_id == parent_id), key=lambda r: r.job.created_at
        )
        return [
            BatchChild(
                job_id=r.job.id,
                input_path=str(r.job.input_payload.get("input_path", "")),
                status=r.status,
                files=self._view(r).files,
                error=r.error,
            )
            for r in children
        ]

    async def open_batch_parent_ids(self) -> list[JobId]:
        """Return non-terminal batch parents."""
        return [
            r.job.id
            for r in self._db.jobs.values()
            if r.job.type is JobType.TRANSCRIBE_BATCH and r.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        ]

    async def set_status(self, job_id: JobId, status: JobStatus, *, now: datetime) -> bool:
        """Apply a manual transition."""
        record = self._transition(job_id, status)
        if record is None:
            return False
        if status is JobStatus.QUEUED:
            record.error, record.max_attempts = None, record.attempt_count + 1
        return True

    async def registration_voices(self, job_ids: list[JobId]) -> dict[JobId, VoiceId]:
        """Return the voice of each register_voice job among job_ids."""
        voices: dict[JobId, VoiceId] = {}
        for job_id in job_ids:
            record = self._db.jobs.get(job_id)
            if record is not None and record.job.type is JobType.REGISTER_VOICE:
                voices[job_id] = VoiceId(record.job.input_payload["voice_id"])
        return voices

    async def owner_of(self, job_id: JobId) -> Caller | None:
        """Return the job's owner."""
        record = self._db.jobs.get(job_id)
        return record.job.owner if record else None

    async def demand(self, *, now: datetime, heartbeat_before: datetime) -> JobDemand:
        """Count runnable and freshly running non-parent jobs."""
        runnable = running = 0
        for record in self._db.jobs.values():
            if record.job.type is JobType.TRANSCRIBE_BATCH:
                continue
            stale = record.heartbeat_at is None or record.heartbeat_at < heartbeat_before
            if record.status is JobStatus.RUNNING and not stale:
                running += 1
            elif (
                record.status is JobStatus.QUEUED
                or (record.status is JobStatus.RETRY_WAIT and record.next_attempt_at and record.next_attempt_at <= now)
                or record.status is JobStatus.RUNNING
            ):
                runnable += 1
        return JobDemand(runnable=runnable, running=running)

    async def release(self, job_id: JobId, *, claim_token: ClaimToken) -> bool:
        """Requeue a job this claim holds and give back its attempt."""
        record = self._held(job_id, claim_token)
        if record is None:
            return False
        record.status, record.lease_owner, record.heartbeat_at = JobStatus.QUEUED, None, None
        record.attempt_count = max(record.attempt_count - 1, 0)
        record.phase, record.progress = None, 0
        return True


class MockJobEventRepository:
    """In-memory JobEventRepository."""

    def __init__(self, db: Database) -> None:
        """Bind to the shared tables."""
        self._db = db

    async def add(self, event: JobEvent) -> None:
        """Append an event."""
        self._db.events.append(event)


class MockAssetRepository:
    """In-memory AssetRepository."""

    def __init__(self, db: Database) -> None:
        """Bind to the shared tables."""
        self._db = db

    async def add(
        self,
        asset: PublishedAsset,
        *,
        owner: Caller,
        job_id: JobId | None,
        voice_version_id: VoiceVersionId | None,
        created_at: datetime,
    ) -> None:
        """Record an asset once per location."""
        if any(a.asset.stored.location == asset.stored.location for a in self._db.assets):
            return
        self._db.assets.append(AssetRecord(asset, owner, job_id, voice_version_id))

    async def locations_for_voice(self, voice_id: VoiceId) -> list[AssetLocation]:
        """Return locations of the voice owner's assets attached to any version of the voice."""
        return sorted((a.asset.stored.location for a in self._voice_assets(voice_id)), key=lambda loc: loc.uri)

    async def locations_for_job(self, job_id: JobId) -> list[AssetLocation]:
        """Return locations of a job's assets."""
        return [a.asset.stored.location for a in self._db.assets if a.job_id == job_id]

    async def known_uris(self, uris: list[str]) -> set[str]:
        """Return URIs among uris that are recorded or are the uploaded input of an unfinished job."""
        recorded = {a.asset.stored.location.uri for a in self._db.assets}
        inputs = (
            UploadedInput.from_payload(record.job.input_payload)
            for record in self._db.jobs.values()
            if not record.status.is_terminal
        )
        pending = {uploaded.stored.location.uri for uploaded in inputs if uploaded is not None}
        return {uri for uri in uris if uri in recorded | pending}

    async def delete_for_voice(self, voice_id: VoiceId) -> int:
        """Delete the rows returned by locations_for_voice."""
        doomed = [id(a) for a in self._voice_assets(voice_id)]
        self._db.assets = [a for a in self._db.assets if id(a) not in doomed]
        return len(doomed)

    def _voice_assets(self, voice_id: VoiceId) -> list[AssetRecord]:
        """Return assets attached to a version of the voice and owned by the voice's owner."""
        voice = self._db.voices.get(voice_id)
        if voice is None:
            return []
        versions = {v.id for v in self._db.versions.values() if v.voice_id == voice_id}
        return [
            a
            for a in self._db.assets
            if a.voice_version_id in versions
            and (a.owner.issuer, a.owner.subject) == (voice.owner.issuer, voice.owner.subject)
        ]


class MockTranscriptRepository:
    """In-memory TranscriptRepository."""

    def __init__(self, db: Database) -> None:
        """Bind to the shared tables."""
        self._db = db

    async def add(self, job_id: JobId, owner: Caller, result: TranscriptionResult, *, created_at: datetime) -> None:
        """Store a transcript."""
        self._db.transcripts[job_id] = result


class MockQualityMetricsRepository:
    """In-memory QualityMetricsRepository."""

    def __init__(self, db: Database) -> None:
        """Bind to the shared tables."""
        self._db = db

    async def add(self, job_id: JobId, metrics: dict[str, Any], *, created_at: datetime) -> None:
        """Store metrics."""
        self._db.metrics[job_id] = metrics


class MockVoiceRepository:
    """In-memory VoiceRepository."""

    def __init__(self, db: Database) -> None:
        """Bind to the shared tables."""
        self._db = db

    async def add(self, voice: NewVoice) -> None:
        """Insert a pending voice and its first version."""
        version = VoiceVersion(
            id=voice.voice_version_id,
            voice_id=voice.voice_id,
            version=1,
            reference_uri=None,
            reference_backend=None,
            reference_sha256=None,
            reference_text=voice.reference_text,
            language=voice.language,
            conditioning_mode=voice.conditioning_mode,
        )
        self._db.versions[version.id] = version
        self._db.voices[voice.voice_id] = Voice(
            id=voice.voice_id,
            owner=voice.owner,
            name=voice.name,
            status=VoiceStatus.PENDING,
            language=voice.language,
            created_at=voice.created_at,
            current_version=version,
        )

    async def get(self, voice_id: VoiceId, owner: Caller | None, *, include_deleted: bool = False) -> Voice | None:
        """Return an owned voice, deleted ones only on request."""
        voice = self._db.voices.get(voice_id)
        if voice is None or (voice.status is VoiceStatus.DELETED and not include_deleted):
            return None
        if owner is not None and (voice.owner.issuer, voice.owner.subject) != (owner.issuer, owner.subject):
            return None
        return replace(voice, current_version=self._latest(voice_id))

    async def list_owned(self, owner: Caller, *, language: str | None) -> list[Voice]:
        """Return the owner's non-deleted voices."""
        voices = [await self.get(voice_id, owner) for voice_id in self._db.voices]
        return [v for v in voices if v is not None and (language is None or v.language == language)]

    async def get_version(self, version_id: VoiceVersionId) -> VoiceVersion | None:
        """Return a version."""
        return self._db.versions.get(version_id)

    async def mark_ready(
        self, version_id: VoiceVersionId, *, reference: AssetLocation, sha256: str, now: datetime
    ) -> bool:
        """Mark a pending voice ready and attach the reference; False if the voice is not pending."""
        version = self._db.versions[version_id]
        voice = self._db.voices[version.voice_id]
        if voice.status is not VoiceStatus.PENDING:
            return False
        version = replace(
            version, reference_uri=reference.uri, reference_backend=reference.backend.value, reference_sha256=sha256
        )
        self._db.versions[version_id] = version
        self._db.voices[voice.id] = replace(voice, status=VoiceStatus.READY, current_version=version)
        return True

    async def mark_failed(self, voice_id: VoiceId, *, now: datetime) -> bool:
        """Mark a pending voice failed."""
        return self._move(voice_id, (VoiceStatus.PENDING,), VoiceStatus.FAILED)

    async def reset_to_pending(self, voice_id: VoiceId, *, now: datetime) -> bool:
        """Move a failed voice back to pending."""
        return self._move(voice_id, (VoiceStatus.FAILED,), VoiceStatus.PENDING)

    async def mark_deleted(self, voice_id: VoiceId, *, now: datetime) -> bool:
        """Mark a non-pending voice deleted and clear references."""
        if not self._move(voice_id, (VoiceStatus.READY, VoiceStatus.FAILED, VoiceStatus.DELETED), VoiceStatus.DELETED):
            return False
        for version_id, version in list(self._db.versions.items()):
            if version.voice_id == voice_id:
                self._db.versions[version_id] = replace(
                    version, reference_uri=None, reference_backend=None, reference_sha256=None, reference_text=None
                )
        return True

    def _move(self, voice_id: VoiceId, sources: tuple[VoiceStatus, ...], target: VoiceStatus) -> bool:
        """Change the status only from one of sources."""
        voice = self._db.voices.get(voice_id)
        if voice is None or voice.status not in sources:
            return False
        self._db.voices[voice_id] = replace(voice, status=target)
        return True

    def _latest(self, voice_id: VoiceId) -> VoiceVersion | None:
        """Return the newest version of a voice."""
        versions = [v for v in self._db.versions.values() if v.voice_id == voice_id]
        return max(versions, key=lambda v: v.version) if versions else None


class MockWorkerRepository:
    """In-memory WorkerRepository."""

    def __init__(self, db: Database) -> None:
        """Bind to the shared tables."""
        self._db = db

    async def register(self, record: WorkerRecord) -> None:
        """Insert or replace a worker's row."""
        self._db.workers[record.id] = record

    async def beat(
        self,
        worker_id: str,
        *,
        status: WorkerStatus,
        now: datetime,
        last_activity_at: datetime | None,
        capabilities: CapabilitySnapshot | None = None,
    ) -> WorkerRecord | None:
        """Update status, heartbeat, activity and optionally capabilities."""
        record = self._db.workers.get(worker_id)
        if record is None:
            return None
        record = replace(
            record,
            status=status,
            heartbeat_at=now,
            last_activity_at=last_activity_at,
            capabilities=capabilities or record.capabilities,
        )
        self._db.workers[worker_id] = record
        return record

    async def mark_stopped(self, worker_id: str, *, now: datetime) -> None:
        """Set status stopped with drained_at."""
        record = self._db.workers.get(worker_id)
        if record is not None:
            self._db.workers[worker_id] = replace(record, status=WorkerStatus.STOPPED, drained_at=now, heartbeat_at=now)

    async def get(self, worker_id: str) -> WorkerRecord | None:
        """Return one worker's row."""
        return self._db.workers.get(worker_id)

    async def latest(self) -> WorkerRecord | None:
        """Return the row with the newest heartbeat."""
        return max(self._db.workers.values(), key=lambda record: record.heartbeat_at, default=None)

    async def active(self, *, heartbeat_after: datetime) -> WorkerRecord | None:
        """Return the newest-started non-stopped worker with a fresh heartbeat."""
        fresh = [
            record
            for record in self._db.workers.values()
            if record.heartbeat_at > heartbeat_after and record.status is not WorkerStatus.STOPPED
        ]
        return max(fresh, key=lambda record: record.started_at, default=None)

    async def request_drain(self, worker_id: str, *, now: datetime) -> bool:
        """Set drain_requested_at if unset."""
        record = self._db.workers.get(worker_id)
        if record is None or record.drain_requested_at is not None:
            return False
        self._db.workers[worker_id] = replace(record, drain_requested_at=now)
        return True

    async def cancel_drain(self, worker_id: str) -> bool:
        """Clear drain_requested_at."""
        record = self._db.workers.get(worker_id)
        if record is None or record.drain_requested_at is None:
            return False
        self._db.workers[worker_id] = replace(record, drain_requested_at=None)
        return True

    async def delete_stopped(self, *, before: datetime) -> int:
        """Delete old stopped rows, always keeping the newest row."""
        newest = await self.latest()
        old = [
            worker_id
            for worker_id, record in self._db.workers.items()
            if record.status is WorkerStatus.STOPPED and record.heartbeat_at < before and record is not newest
        ]
        for worker_id in old:
            del self._db.workers[worker_id]
        return len(old)
