"""Repository ports, typed with domain types only."""

from datetime import datetime
from typing import Any, Protocol

from app.domain.assets import AssetLocation, PublishedAsset
from app.domain.caller import Caller
from app.domain.ids import ClaimToken, JobId, VoiceId, VoiceVersionId
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
    JobView,
    NewJob,
)
from app.domain.stt import TranscriptionResult
from app.domain.voices import NewVoice, Voice, VoiceVersion
from app.domain.workers import CapabilitySnapshot, JobDemand, WorkerRecord, WorkerStatus


class JobRepository(Protocol):
    """Persistence of jobs, including atomic claims and conditional transitions."""

    async def add(self, job: NewJob) -> None:
        """Insert a queued job."""
        ...

    async def find_by_request_id(self, owner: Caller, request_id: str) -> JobId | None:
        """Return the job previously submitted by this owner with this request_id."""
        ...

    async def get(self, job_id: JobId, owner: Caller | None) -> JobView | None:
        """Return the job if it exists and is owned by owner; owner None is for trusted CLI use only."""
        ...

    async def page(self, owner: Caller, job_filter: JobFilter, *, limit: int, cursor: JobCursor | None) -> JobPage:
        """Return one keyset-paginated page of the owner's jobs, newest first."""
        ...

    async def claim(self, job_id: JobId, *, lease_owner: str, now: datetime) -> ClaimedJob | None:
        """Atomically move a queued job to running under a new claim token; None if it is not claimable."""
        ...

    async def set_phase(
        self, job_id: JobId, *, claim_token: ClaimToken, phase: JobPhase, progress: int, now: datetime
    ) -> bool:
        """Record the phase and progress of a job this claim holds; return whether it is still held."""
        ...

    async def heartbeat(self, job_id: JobId, *, claim_token: ClaimToken, now: datetime) -> bool:
        """Refresh heartbeat_at of a job this claim holds; return whether it is still held."""
        ...

    async def complete(
        self, job_id: JobId, completion: JobCompletion, *, claim_token: ClaimToken, now: datetime
    ) -> bool:
        """Move a job this claim holds to the completion status with its result; return whether it transitioned."""
        ...

    async def fail(self, job_id: JobId, error: JobError, *, claim_token: ClaimToken, now: datetime) -> bool:
        """Move a job this claim holds to failed; return whether it transitioned."""
        ...

    async def schedule_retry(
        self, job_id: JobId, error: JobError, *, claim_token: ClaimToken, next_attempt_at: datetime
    ) -> bool:
        """Move a job this claim holds to retry_wait; return whether it transitioned."""
        ...

    async def requeue_due(self, *, now: datetime) -> int:
        """Move retry_wait jobs whose next attempt is due back to queued; return the count."""
        ...

    async def queued_ids(self, *, limit: int) -> list[JobId]:
        """Return queued, non-parent job IDs, oldest first."""
        ...

    async def interrupt_stale(self, *, heartbeat_before: datetime) -> list[JobId]:
        """Mark running jobs with an expired heartbeat as interrupted; return their IDs."""
        ...

    async def resolve_interrupted(self, *, now: datetime) -> tuple[list[JobId], list[JobId]]:
        """Requeue interrupted jobs with attempts left and fail the rest; return (requeued, failed)."""
        ...

    async def batch_counts(self, parent_id: JobId) -> BatchCounts:
        """Return the child status counts of a batch parent."""
        ...

    async def update_batch_parent(
        self, parent_id: JobId, *, counts: BatchCounts, status: JobStatus | None, now: datetime
    ) -> bool:
        """Store aggregated counts and, if given, the parent's new status; return whether the status changed."""
        ...

    async def batch_children(self, parent_id: JobId) -> list[BatchChild]:
        """Return every child of a batch parent in creation order."""
        ...

    async def open_batch_parent_ids(self) -> list[JobId]:
        """Return batch parents that are not yet terminal."""
        ...

    async def set_status(self, job_id: JobId, status: JobStatus, *, now: datetime) -> bool:
        """Apply an allowed manual transition such as a CLI cancel or retry; return whether it applied."""
        ...

    async def registration_voices(self, job_ids: list[JobId]) -> dict[JobId, VoiceId]:
        """Return the voice of each register_voice job among job_ids; other job types are left out."""
        ...

    async def owner_of(self, job_id: JobId) -> Caller | None:
        """Return the owner of a job, for trusted internal use."""
        ...

    async def demand(self, *, now: datetime, heartbeat_before: datetime) -> JobDemand:
        """Count runnable jobs (queued, due retries, stale running) and running jobs with a fresh heartbeat."""
        ...

    async def release(self, job_id: JobId, *, claim_token: ClaimToken) -> bool:
        """Return a job this claim holds to queued without using up its attempt; return whether it did."""
        ...


class JobEventRepository(Protocol):
    """Append-only job history."""

    async def add(self, event: JobEvent) -> None:
        """Append one event."""
        ...


class AssetRepository(Protocol):
    """Metadata rows for assets whose bytes live in an AssetStore."""

    async def add(
        self,
        asset: PublishedAsset,
        *,
        owner: Caller,
        job_id: JobId | None,
        voice_version_id: VoiceVersionId | None,
        created_at: datetime,
    ) -> None:
        """Record one published asset."""
        ...

    async def locations_for_voice(self, voice_id: VoiceId) -> list[AssetLocation]:
        """Return every stored recording of every version of a voice that belongs to the voice's owner."""
        ...

    async def locations_for_job(self, job_id: JobId) -> list[AssetLocation]:
        """Return the locations of every asset of a job."""
        ...

    async def delete_for_voice(self, voice_id: VoiceId) -> int:
        """Delete the asset rows returned by locations_for_voice; return the count."""
        ...

    async def known_uris(self, uris: list[str]) -> set[str]:
        """Return which of the given URIs are recorded or are the uploaded input of an unfinished job."""
        ...


class TranscriptRepository(Protocol):
    """Transcripts with segments and words."""

    async def add(self, job_id: JobId, owner: Caller, result: TranscriptionResult, *, created_at: datetime) -> None:
        """Store a transcript with its segments and words."""
        ...


class QualityMetricsRepository(Protocol):
    """Per-job quality metrics."""

    async def add(self, job_id: JobId, metrics: dict[str, Any], *, created_at: datetime) -> None:
        """Store the metrics of one job."""
        ...


class VoiceRepository(Protocol):
    """Owned voices and immutable voice versions."""

    async def add(self, voice: NewVoice) -> None:
        """Insert a pending voice with its first version."""
        ...

    async def get(self, voice_id: VoiceId, owner: Caller | None, *, include_deleted: bool = False) -> Voice | None:
        """Return a voice owned by owner; deleted voices only on request, and owner None only for trusted CLI use."""
        ...

    async def list_owned(self, owner: Caller, *, language: str | None) -> list[Voice]:
        """Return the owner's non-deleted voices."""
        ...

    async def get_version(self, version_id: VoiceVersionId) -> VoiceVersion | None:
        """Return a voice version."""
        ...

    async def mark_ready(
        self, version_id: VoiceVersionId, *, reference: AssetLocation, sha256: str, now: datetime
    ) -> bool:
        """Mark the version's pending voice ready and attach the reference; False if the voice is not pending."""
        ...

    async def mark_failed(self, voice_id: VoiceId, *, now: datetime) -> bool:
        """Mark a pending voice failed; return whether it changed."""
        ...

    async def reset_to_pending(self, voice_id: VoiceId, *, now: datetime) -> bool:
        """Move a failed voice back to pending for a manually retried registration; return whether it changed."""
        ...

    async def mark_deleted(self, voice_id: VoiceId, *, now: datetime) -> bool:
        """Mark a non-pending voice deleted and clear its version references; False for a pending voice."""
        ...


class WorkerRepository(Protocol):
    """Registry of GPU worker processes, their status, heartbeats, capabilities and drain requests."""

    async def register(self, record: WorkerRecord) -> None:
        """Insert or replace a worker's row."""
        ...

    async def beat(
        self,
        worker_id: str,
        *,
        status: WorkerStatus,
        now: datetime,
        last_activity_at: datetime | None,
        capabilities: CapabilitySnapshot | None = None,
    ) -> WorkerRecord | None:
        """Record status, heartbeat, activity and optionally capabilities; return the row, None if it is gone."""
        ...

    async def mark_stopped(self, worker_id: str, *, now: datetime) -> None:
        """Set status stopped with drained_at and a final heartbeat."""
        ...

    async def get(self, worker_id: str) -> WorkerRecord | None:
        """Return one worker's row."""
        ...

    async def latest(self) -> WorkerRecord | None:
        """Return the row with the newest heartbeat, whatever its status."""
        ...

    async def active(self, *, heartbeat_after: datetime) -> WorkerRecord | None:
        """Return the most recently started worker whose heartbeat is newer than heartbeat_after."""
        ...

    async def request_drain(self, worker_id: str, *, now: datetime) -> bool:
        """Set drain_requested_at if unset; return whether a row changed."""
        ...

    async def cancel_drain(self, worker_id: str) -> bool:
        """Clear drain_requested_at; return whether a row changed."""
        ...

    async def delete_stopped(self, *, before: datetime) -> int:
        """Delete stopped rows older than before, always keeping the newest row; return the count."""
        ...
