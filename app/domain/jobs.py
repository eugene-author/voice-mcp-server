"""Job types, lifecycle state machine, commands and job projections."""

import base64
import binascii
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from app.domain.assets import AssetRole, PublishedAsset
from app.domain.caller import Caller
from app.domain.errors import InternalError, InvalidRequestError
from app.domain.ids import ClaimToken, JobId, VoiceId, VoiceVersionId
from app.domain.stt import ResolvedSttOptions, SttOptions, Substitution, TranscriptionResult
from app.domain.tts import ResolvedTtsOptions, TtsOptions


class JobType(StrEnum):
    """Kind of work a job performs."""

    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    TRANSCRIBE_BATCH = "transcribe_batch"
    REGISTER_VOICE = "register_voice"


class JobStatus(StrEnum):
    """Lifecycle state of a job."""

    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Return whether no further automatic transition happens."""
        return self in _TERMINAL

    def can_transition_to(self, target: JobStatus) -> bool:
        """Return whether the state machine allows moving from this status to target."""
        return target in _TRANSITIONS[self]


_TERMINAL = frozenset({JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_ERRORS, JobStatus.FAILED, JobStatus.CANCELLED})

_TRANSITIONS: Mapping[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED}),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.COMPLETED,
            JobStatus.COMPLETED_WITH_ERRORS,
            JobStatus.FAILED,
            JobStatus.RETRY_WAIT,
            JobStatus.INTERRUPTED,
        }
    ),
    JobStatus.RETRY_WAIT: frozenset({JobStatus.QUEUED, JobStatus.CANCELLED}),
    JobStatus.INTERRUPTED: frozenset({JobStatus.QUEUED, JobStatus.FAILED}),
    JobStatus.FAILED: frozenset({JobStatus.QUEUED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.COMPLETED_WITH_ERRORS: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}
"""Allowed transitions; failed -> queued is the explicit CLI retry."""


class JobPhase(StrEnum):
    """Processing phase of a running job."""

    DECODING = "decoding"
    DENOISING = "denoising"
    VAD = "vad"
    INFERENCING = "inferencing"
    EXPORTING = "exporting"
    PUBLISHING = "publishing"


@dataclass(frozen=True, slots=True)
class TranscribeCommand:
    """Request to transcribe one input file."""

    input_path: Path
    model: str | None = None
    language: str | None = None
    preset: str | None = None
    options: SttOptions | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class SynthesizeCommand:
    """Request to synthesize text in a voice."""

    text: str
    voice_id: str
    language: str | None = None
    preset: str | None = None
    options: TtsOptions | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class BatchTranscribeCommand:
    """Request to transcribe every matching file in a directory."""

    directory: Path
    recursive: bool = False
    preset: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class RegisterVoiceCommand:
    """Request to register a cloned voice from an authorized reference recording."""

    name: str
    reference_audio_path: Path
    consent_confirmed: bool
    reference_text: str | None = None
    language: str | None = None
    request_id: str | None = None


type JobCommand = TranscribeCommand | SynthesizeCommand | BatchTranscribeCommand | RegisterVoiceCommand
"""Any command that creates a job."""


@dataclass(frozen=True, slots=True)
class TranscriptionSpec:
    """Frozen, fully resolved description of a transcription job."""

    input_path: str
    model: str
    language: str | None
    options: ResolvedSttOptions
    preserve_input: bool = True

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON input payload stored with the job."""
        return {
            "input_path": self.input_path,
            "model": self.model,
            "language": self.language,
            "options": self.options.to_dict(),
            "preserve_input": self.preserve_input,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild the spec from a stored payload."""
        return cls(
            input_path=payload["input_path"],
            model=payload["model"],
            language=payload.get("language"),
            options=ResolvedSttOptions.from_dict(payload["options"]),
            preserve_input=payload.get("preserve_input", True),
        )


@dataclass(frozen=True, slots=True)
class SynthesisSpec:
    """Frozen description of a synthesis job with its pinned voice version."""

    text: str
    language: str
    voice_id: str
    voice_version_id: VoiceVersionId | None
    """Pinned version for cloned voices; None for preset speakers."""

    preset_speaker: str | None
    options: ResolvedTtsOptions

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON input payload stored with the job."""
        return {
            "text": self.text,
            "language": self.language,
            "voice_id": self.voice_id,
            "voice_version_id": str(self.voice_version_id) if self.voice_version_id else None,
            "preset_speaker": self.preset_speaker,
            "options": self.options.to_dict(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild the spec from a stored payload."""
        version = payload.get("voice_version_id")
        return cls(
            text=payload["text"],
            language=payload["language"],
            voice_id=payload["voice_id"],
            voice_version_id=VoiceVersionId(uuid.UUID(version)) if version else None,
            preset_speaker=payload.get("preset_speaker"),
            options=ResolvedTtsOptions.from_dict(payload["options"]),
        )


@dataclass(frozen=True, slots=True)
class VoiceRegistrationSpec:
    """Frozen description of a voice registration job."""

    voice_id: VoiceId
    voice_version_id: VoiceVersionId
    version: int
    reference_audio_path: str
    reference_text: str | None
    language: str | None
    denoise: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON input payload stored with the job."""
        return {
            "voice_id": self.voice_id,
            "voice_version_id": str(self.voice_version_id),
            "version": self.version,
            "reference_audio_path": self.reference_audio_path,
            "reference_text": self.reference_text,
            "language": self.language,
            "denoise": self.denoise,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild the spec from a stored payload."""
        return cls(
            voice_id=VoiceId(payload["voice_id"]),
            voice_version_id=VoiceVersionId(uuid.UUID(payload["voice_version_id"])),
            version=payload["version"],
            reference_audio_path=payload["reference_audio_path"],
            reference_text=payload.get("reference_text"),
            language=payload.get("language"),
            denoise=payload.get("denoise", False),
        )


@dataclass(frozen=True, slots=True)
class NewJob:
    """Job ready to be persisted in the queued state."""

    id: JobId
    type: JobType
    owner: Caller
    input_payload: dict[str, Any]
    config_snapshot: dict[str, Any]
    created_at: datetime
    max_attempts: int
    request_id: str | None = None
    parent_job_id: JobId | None = None
    substitutions: tuple[Substitution, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    """Job claimed by the worker for execution."""

    id: JobId
    type: JobType
    owner: Caller
    input_payload: dict[str, Any]
    config_snapshot: dict[str, Any]
    attempt_count: int
    max_attempts: int
    claim_token: ClaimToken
    """New for every claim, unlike attempt_count (a drain refunds it) and lease_owner (a process reclaims with it)."""

    parent_job_id: JobId | None = None
    substitutions: tuple[Substitution, ...] = ()

    @property
    def has_attempts_left(self) -> bool:
        """Return whether a retry is allowed after this attempt."""
        return self.attempt_count < self.max_attempts


@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    """Execution environment actually used by a job."""

    backend: str
    model_id: str
    device: str
    compute_type: str | None = None
    model_revision: str | None = None


@dataclass(frozen=True, slots=True)
class JobCompletion:
    """Successful handler result to record when completing a job."""

    result: dict[str, Any] = field(default_factory=dict)
    """Compact result payload such as a text preview."""

    assets: tuple[PublishedAsset, ...] = ()
    warnings: tuple[str, ...] = ()
    execution: ExecutionInfo | None = None
    transcript: TranscriptionResult | None = None
    metrics: dict[str, Any] | None = None
    status: JobStatus = JobStatus.COMPLETED
    voice_version_id: VoiceVersionId | None = None
    """Set by voice registration: every asset (processed reference and preserved original) is linked to this version
    so deleting the voice purges all of them, and the version's voice becomes ready."""

    @property
    def voice_reference(self) -> PublishedAsset:
        """Return the processed reference recording of a registration completion."""
        for asset in self.assets:
            if asset.role is AssetRole.VOICE_REFERENCE:
                return asset
        raise InternalError("a voice registration completion has no voice_reference asset")


@dataclass(frozen=True, slots=True)
class BatchCounts:
    """Aggregated child statuses of a batch parent."""

    total: int
    completed: int
    failed: int
    queued: int

    @property
    def finished(self) -> int:
        """Return the number of children in a terminal state."""
        return self.completed + self.failed

    def final_status(self) -> JobStatus | None:
        """Return the parent's terminal status once every child finished, else None."""
        if self.finished < self.total:
            return None
        if self.failed == 0:
            return JobStatus.COMPLETED
        return JobStatus.FAILED if self.completed == 0 else JobStatus.COMPLETED_WITH_ERRORS

    def to_dict(self) -> dict[str, int]:
        """Return the public counts object."""
        return {"total": self.total, "completed": self.completed, "failed": self.failed, "queued": self.queued}


@dataclass(frozen=True, slots=True)
class BatchChild:
    """One child of a batch, as listed in its manifest."""

    job_id: JobId
    input_path: str
    status: JobStatus
    files: dict[str, str]
    error: JobError | None = None


@dataclass(frozen=True, slots=True)
class JobError:
    """Recorded failure of a job."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class JobView:
    """Read projection of one job for its owner."""

    id: JobId
    type: JobType
    status: JobStatus
    created_at: datetime
    progress: int = 0
    phase: JobPhase | None = None
    files: dict[str, str] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: JobError | None = None
    parent_job_id: JobId | None = None
    counts: BatchCounts | None = None
    voice_id: str | None = None


@dataclass(frozen=True, slots=True)
class JobFilter:
    """Optional list_jobs filters."""

    status: JobStatus | None = None
    type: JobType | None = None
    parent_job_id: JobId | None = None


@dataclass(frozen=True, slots=True)
class JobCursor:
    """Keyset pagination position: the last returned row's created_at and id."""

    created_at: datetime
    job_id: JobId

    def encode(self) -> str:
        """Return an opaque URL-safe cursor string."""
        raw = json.dumps({"t": self.created_at.isoformat(), "id": str(self.job_id)}).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, cursor: str) -> Self:
        """Parse a cursor string or raise InvalidRequestError."""
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            data = json.loads(base64.urlsafe_b64decode(padded))
            return cls(created_at=datetime.fromisoformat(data["t"]), job_id=JobId(uuid.UUID(data["id"])))
        except (ValueError, KeyError, TypeError, binascii.Error) as exc:
            raise InvalidRequestError("invalid cursor") from exc


@dataclass(frozen=True, slots=True)
class JobPage:
    """One page of job summaries."""

    items: tuple[JobView, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class JobEvent:
    """Append-only job history entry."""

    job_id: JobId
    at: datetime
    event_type: str
    phase: JobPhase | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


type ProgressCallback = Callable[[JobPhase, int], Awaitable[None]]
"""Records a running job's phase and progress percentage."""
