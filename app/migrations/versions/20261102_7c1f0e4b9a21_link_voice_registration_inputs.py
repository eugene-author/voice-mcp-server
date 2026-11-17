"""Link preserved registration originals to their voice version so voice deletion purges them.

Before this revision only the processed voice_reference asset of a register_voice job was attached to the voice
version; the preserved original (role ``input``) was recorded with the job alone and survived voice deletion.

The backfill only follows verified relationships: the asset belongs to a register_voice job, the job payload names a
voice and a well-formed version UUID, that version exists and belongs to that voice, and the job, voice and asset all
have the same owner. Rows with absent or malformed payload fields are left unchanged. Transcription and synthesis
assets are not touched because their jobs are not register_voice jobs.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "7c1f0e4b9a21"
down_revision: str | None = "2d05911d2db1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID_PATTERN = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

BACKFILL = f"""
UPDATE media_assets AS asset
SET voice_version_id = version.id
FROM jobs AS job
JOIN voice_versions AS version
  ON version.id = CASE
       WHEN job.input_payload ->> 'voice_version_id' ~ '{UUID_PATTERN}'
       THEN (job.input_payload ->> 'voice_version_id')::uuid
     END
JOIN voices AS voice
  ON voice.id = version.voice_id
WHERE asset.job_id = job.id
  AND asset.voice_version_id IS NULL
  AND asset.role = 'input'
  AND job.type = 'register_voice'
  AND voice.id = job.input_payload ->> 'voice_id'
  AND voice.owner_issuer = job.owner_issuer AND voice.owner_subject = job.owner_subject
  AND asset.owner_issuer = job.owner_issuer AND asset.owner_subject = job.owner_subject
"""


def upgrade() -> None:
    """Attach unlinked registration originals to the voice version their job registered."""
    op.execute(BACKFILL)


def downgrade() -> None:
    """Leave the links in place: removing them would hide recordings from voice deletion again."""
