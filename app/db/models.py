"""SQLAlchemy ORM models; they stay inside app/db/."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.domain.assets import AssetRole, StorageBackend
from app.domain.jobs import JobPhase, JobStatus, JobType
from app.domain.tts import ConditioningMode
from app.domain.voices import VoiceStatus
from app.domain.workers import WorkerStatus


def _in_check(column: str, values: list[str]) -> str:
    """Return a SQL check expression restricting a column to string values."""
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({quoted})"


class JobRow(Base):
    """Persistent job."""

    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(_in_check("status", [s.value for s in JobStatus]), name="status"),
        CheckConstraint(_in_check("type", [t.value for t in JobType]), name="type"),
        CheckConstraint(_in_check("phase", [p.value for p in JobPhase]), name="phase"),
        CheckConstraint("progress BETWEEN 0 AND 100", name="progress"),
        Index("ix_jobs_owner_created", "owner_issuer", "owner_subject", text("created_at DESC"), text("id DESC")),
        Index("ix_jobs_status_created", "status", text("created_at DESC"), text("id DESC")),
        Index("ix_jobs_type_created", "type", text("created_at DESC"), text("id DESC")),
        Index("ix_jobs_parent_created", "parent_job_id", text("created_at DESC"), text("id DESC")),
        Index(
            "uq_jobs_owner_request_id",
            "owner_issuer",
            "owner_subject",
            "request_id",
            unique=True,
            postgresql_where=text("request_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    parent_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE")
    )
    request_id: Mapped[str | None] = mapped_column(String(128))
    owner_issuer: Mapped[str] = mapped_column(String(512))
    owner_subject: Mapped[str] = mapped_column(String(512))
    owner_client_id: Mapped[str | None] = mapped_column(String(512))
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    substitutions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    warnings: Mapped[list[str]] = mapped_column(JSONB, default=list)
    counts: Mapped[dict[str, int] | None] = mapped_column(JSONB)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer)
    phase: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    claim_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    backend: Mapped[str | None] = mapped_column(String(32))
    model_id: Mapped[str | None] = mapped_column(String(128))
    model_revision: Mapped[str | None] = mapped_column(String(128))
    device: Mapped[str | None] = mapped_column(String(32))
    compute_type: Mapped[str | None] = mapped_column(String(32))


class JobEventRow(Base):
    """Append-only job history entry."""

    __tablename__ = "job_events"
    __table_args__ = (Index("ix_job_events_job_at", "job_id", "at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    event_type: Mapped[str] = mapped_column(String(64))
    phase: Mapped[str | None] = mapped_column(String(32))
    message: Mapped[str | None] = mapped_column(Text)
    event_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class VoiceRow(Base):
    """Owned cloned voice."""

    __tablename__ = "voices"
    __table_args__ = (
        CheckConstraint(_in_check("status", [s.value for s in VoiceStatus]), name="status"),
        Index("ix_voices_owner_created", "owner_issuer", "owner_subject", text("created_at DESC")),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_issuer: Mapped[str] = mapped_column(String(512))
    owner_subject: Mapped[str] = mapped_column(String(512))
    owner_client_id: Mapped[str | None] = mapped_column(String(512))
    name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))
    language: Mapped[str | None] = mapped_column(String(16))
    consent_confirmed: Mapped[bool] = mapped_column(Boolean)
    consent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consent_source: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VoiceVersionRow(Base):
    """Immutable, runtime-portable reference of one voice version."""

    __tablename__ = "voice_versions"
    __table_args__ = (
        UniqueConstraint("voice_id", "version", name="uq_voice_versions_voice_version"),
        CheckConstraint(_in_check("conditioning_mode", [m.value for m in ConditioningMode]), name="mode"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    voice_id: Mapped[str] = mapped_column(String(64), ForeignKey("voices.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    reference_uri: Mapped[str | None] = mapped_column(Text)
    reference_backend: Mapped[str | None] = mapped_column(String(16))
    reference_sha256: Mapped[str | None] = mapped_column(String(64))
    reference_text: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(16))
    conditioning_mode: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MediaAssetRow(Base):
    """Metadata of a stored asset."""

    __tablename__ = "media_assets"
    __table_args__ = (
        CheckConstraint(_in_check("role", [r.value for r in AssetRole]), name="role"),
        CheckConstraint(_in_check("storage_backend", [b.value for b in StorageBackend]), name="backend"),
        UniqueConstraint("uri", name="uq_media_assets_uri"),
        Index("ix_media_assets_job", "job_id"),
        Index("ix_media_assets_voice_version", "voice_version_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"))
    voice_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("voice_versions.id", ondelete="CASCADE")
    )
    owner_issuer: Mapped[str] = mapped_column(String(512))
    owner_subject: Mapped[str] = mapped_column(String(512))
    role: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(64))
    storage_backend: Mapped[str] = mapped_column(String(16))
    uri: Mapped[str] = mapped_column(Text)
    bucket: Mapped[str | None] = mapped_column(String(255))
    object_key: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    content_type: Mapped[str] = mapped_column(String(128))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    sample_rate: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TranscriptRow(Base):
    """Transcript of one STT job."""

    __tablename__ = "transcripts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), unique=True
    )
    owner_issuer: Mapped[str] = mapped_column(String(512))
    owner_subject: Mapped[str] = mapped_column(String(512))
    language: Mapped[str] = mapped_column(String(16))
    language_probability: Mapped[float | None] = mapped_column(Float)
    duration_seconds: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TranscriptSegmentRow(Base):
    """Recognized segment with source-relative timing."""

    __tablename__ = "transcript_segments"
    __table_args__ = (UniqueConstraint("transcript_id", "position", name="uq_transcript_segments_position"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transcript_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transcripts.id", ondelete="CASCADE")
    )
    position: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    no_speech_prob: Mapped[float | None] = mapped_column(Float)
    avg_logprob: Mapped[float | None] = mapped_column(Float)
    compression_ratio: Mapped[float | None] = mapped_column(Float)


class TranscriptWordRow(Base):
    """Recognized word with source-relative timing."""

    __tablename__ = "transcript_words"
    __table_args__ = (Index("ix_transcript_words_segment", "segment_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    segment_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("transcript_segments.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    probability: Mapped[float | None] = mapped_column(Float)


class QualityMetricsRow(Base):
    """Quality and performance metrics of one job."""

    __tablename__ = "quality_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), unique=True
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    wer: Mapped[float | None] = mapped_column(Float)
    cer: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class WorkerRow(Base):
    """Registered GPU worker process."""

    __tablename__ = "workers"
    __table_args__ = (
        CheckConstraint(_in_check("status", [s.value for s in WorkerStatus]), name="status"),
        Index("ix_workers_heartbeat", text("heartbeat_at DESC")),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(String(16))
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSONB)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_activity_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    drain_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    drained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    version: Mapped[str] = mapped_column(Text)
