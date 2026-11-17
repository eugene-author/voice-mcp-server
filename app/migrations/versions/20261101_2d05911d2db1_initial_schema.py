"""Initial schema: jobs, events, assets, transcripts, voices and quality metrics."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "2d05911d2db1"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this revision."""
    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("parent_job_id", sa.UUID(), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("owner_issuer", sa.String(length=512), nullable=False),
        sa.Column("owner_subject", sa.String(length=512), nullable=False),
        sa.Column("owner_client_id", sa.String(length=512), nullable=True),
        sa.Column("input_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("config_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("substitutions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("warnings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("counts", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("phase", sa.String(length=32), nullable=True),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(length=256), nullable=True),
        sa.Column("result_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("backend", sa.String(length=32), nullable=True),
        sa.Column("model_id", sa.String(length=128), nullable=True),
        sa.Column("model_revision", sa.String(length=128), nullable=True),
        sa.Column("device", sa.String(length=32), nullable=True),
        sa.Column("compute_type", sa.String(length=32), nullable=True),
        sa.CheckConstraint(
            "phase IN ('decoding', 'denoising', 'vad', 'inferencing', 'exporting', 'publishing')",
            name=op.f("ck_jobs_phase"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'retry_wait', 'interrupted', 'completed', 'completed_with_errors', 'failed', 'cancelled')",
            name=op.f("ck_jobs_status"),
        ),
        sa.CheckConstraint(
            "type IN ('speech_to_text', 'text_to_speech', 'transcribe_batch', 'register_voice')",
            name=op.f("ck_jobs_type"),
        ),
        sa.CheckConstraint("progress BETWEEN 0 AND 100", name=op.f("ck_jobs_progress")),
        sa.ForeignKeyConstraint(
            ["parent_job_id"], ["jobs.id"], name=op.f("fk_jobs_parent_job_id_jobs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
    )
    op.create_index(
        "ix_jobs_owner_created",
        "jobs",
        ["owner_issuer", "owner_subject", sa.literal_column("created_at DESC"), sa.literal_column("id DESC")],
        unique=False,
    )
    op.create_index(
        "ix_jobs_parent_created",
        "jobs",
        ["parent_job_id", sa.literal_column("created_at DESC"), sa.literal_column("id DESC")],
        unique=False,
    )
    op.create_index(
        "ix_jobs_status_created",
        "jobs",
        ["status", sa.literal_column("created_at DESC"), sa.literal_column("id DESC")],
        unique=False,
    )
    op.create_index(
        "ix_jobs_type_created",
        "jobs",
        ["type", sa.literal_column("created_at DESC"), sa.literal_column("id DESC")],
        unique=False,
    )
    op.create_index(
        "uq_jobs_owner_request_id",
        "jobs",
        ["owner_issuer", "owner_subject", "request_id"],
        unique=True,
        postgresql_where=sa.text("request_id IS NOT NULL"),
    )
    op.create_table(
        "voices",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("owner_issuer", sa.String(length=512), nullable=False),
        sa.Column("owner_subject", sa.String(length=512), nullable=False),
        sa.Column("owner_client_id", sa.String(length=512), nullable=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("consent_confirmed", sa.Boolean(), nullable=False),
        sa.Column("consent_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consent_source", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'ready', 'failed', 'deleted')", name=op.f("ck_voices_status")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_voices")),
    )
    op.create_index(
        "ix_voices_owner_created",
        "voices",
        ["owner_issuer", "owner_subject", sa.literal_column("created_at DESC")],
        unique=False,
    )
    op.create_table(
        "job_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("phase", sa.String(length=32), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name=op.f("fk_job_events_job_id_jobs"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job_events")),
    )
    op.create_index("ix_job_events_job_at", "job_events", ["job_id", "at"], unique=False)
    op.create_table(
        "quality_metrics",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("wer", sa.Float(), nullable=True),
        sa.Column("cer", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name=op.f("fk_quality_metrics_job_id_jobs"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_quality_metrics")),
        sa.UniqueConstraint("job_id", name=op.f("uq_quality_metrics_job_id")),
    )
    op.create_table(
        "transcripts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column("owner_issuer", sa.String(length=512), nullable=False),
        sa.Column("owner_subject", sa.String(length=512), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("language_probability", sa.Float(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("cleaned_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name=op.f("fk_transcripts_job_id_jobs"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transcripts")),
        sa.UniqueConstraint("job_id", name=op.f("uq_transcripts_job_id")),
    )
    op.create_table(
        "voice_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("voice_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("reference_uri", sa.Text(), nullable=True),
        sa.Column("reference_backend", sa.String(length=16), nullable=True),
        sa.Column("reference_sha256", sa.String(length=64), nullable=True),
        sa.Column("reference_text", sa.Text(), nullable=True),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("conditioning_mode", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("conditioning_mode IN ('icl', 'xvector')", name=op.f("ck_voice_versions_mode")),
        sa.ForeignKeyConstraint(
            ["voice_id"], ["voices.id"], name=op.f("fk_voice_versions_voice_id_voices"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_voice_versions")),
        sa.UniqueConstraint("voice_id", "version", name="uq_voice_versions_voice_version"),
    )
    op.create_table(
        "media_assets",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=True),
        sa.Column("voice_version_id", sa.UUID(), nullable=True),
        sa.Column("owner_issuer", sa.String(length=512), nullable=False),
        sa.Column("owner_subject", sa.String(length=512), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("storage_backend", sa.String(length=16), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("bucket", sa.String(length=255), nullable=True),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("sample_rate", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "role IN ('input', 'working', 'output', 'voice_reference', 'manifest')", name=op.f("ck_media_assets_role")
        ),
        sa.CheckConstraint("storage_backend IN ('local', 's3')", name=op.f("ck_media_assets_backend")),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name=op.f("fk_media_assets_job_id_jobs"), ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["voice_version_id"],
            ["voice_versions.id"],
            name=op.f("fk_media_assets_voice_version_id_voice_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_media_assets")),
        sa.UniqueConstraint("uri", name="uq_media_assets_uri"),
    )
    op.create_index("ix_media_assets_job", "media_assets", ["job_id"], unique=False)
    op.create_index("ix_media_assets_voice_version", "media_assets", ["voice_version_id"], unique=False)
    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("transcript_id", sa.UUID(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("start", sa.Float(), nullable=False),
        sa.Column("end", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("no_speech_prob", sa.Float(), nullable=True),
        sa.Column("avg_logprob", sa.Float(), nullable=True),
        sa.Column("compression_ratio", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["transcript_id"],
            ["transcripts.id"],
            name=op.f("fk_transcript_segments_transcript_id_transcripts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transcript_segments")),
        sa.UniqueConstraint("transcript_id", "position", name="uq_transcript_segments_position"),
    )
    op.create_table(
        "transcript_words",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start", sa.Float(), nullable=False),
        sa.Column("end", sa.Float(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["transcript_segments.id"],
            name=op.f("fk_transcript_words_segment_id_transcript_segments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transcript_words")),
    )
    op.create_index("ix_transcript_words_segment", "transcript_words", ["segment_id"], unique=False)


def downgrade() -> None:
    """Revert this revision."""
    op.drop_index("ix_transcript_words_segment", table_name="transcript_words")
    op.drop_table("transcript_words")
    op.drop_table("transcript_segments")
    op.drop_index("ix_media_assets_voice_version", table_name="media_assets")
    op.drop_index("ix_media_assets_job", table_name="media_assets")
    op.drop_table("media_assets")
    op.drop_table("voice_versions")
    op.drop_table("transcripts")
    op.drop_table("quality_metrics")
    op.drop_index("ix_job_events_job_at", table_name="job_events")
    op.drop_table("job_events")
    op.drop_index("ix_voices_owner_created", table_name="voices")
    op.drop_table("voices")
    op.drop_index("uq_jobs_owner_request_id", table_name="jobs", postgresql_where=sa.text("request_id IS NOT NULL"))
    op.drop_index("ix_jobs_type_created", table_name="jobs")
    op.drop_index("ix_jobs_status_created", table_name="jobs")
    op.drop_index("ix_jobs_parent_created", table_name="jobs")
    op.drop_index("ix_jobs_owner_created", table_name="jobs")
    op.drop_table("jobs")
