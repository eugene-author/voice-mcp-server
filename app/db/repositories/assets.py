"""SQLAlchemy media asset repository; the bytes themselves live in an AssetStore."""

import uuid
from datetime import datetime

from sqlalchemy import Select, delete, select, union
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import JobRow, MediaAssetRow, VoiceRow, VoiceVersionRow
from app.domain.assets import INPUT_LOCATION_FIELD, AssetLocation, PublishedAsset, StorageBackend
from app.domain.caller import Caller
from app.domain.ids import JobId, VoiceId, VoiceVersionId, new_asset_id
from app.domain.jobs import JobStatus

TERMINAL_STATUSES = tuple(status.value for status in JobStatus if status.is_terminal)


def _voice_asset_ids(voice_id: VoiceId) -> Select[uuid.UUID]:
    """Select the assets attached to any version of a voice and owned by the voice's owner."""
    return (
        select(MediaAssetRow.id)
        .join(VoiceVersionRow, VoiceVersionRow.id == MediaAssetRow.voice_version_id)
        .join(VoiceRow, VoiceRow.id == VoiceVersionRow.voice_id)
        .where(
            VoiceRow.id == voice_id,
            MediaAssetRow.owner_issuer == VoiceRow.owner_issuer,
            MediaAssetRow.owner_subject == VoiceRow.owner_subject,
        )
    )


class SqlAssetRepository:
    """AssetRepository over media_assets."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to a session owned by the unit of work."""
        self._session = session

    async def add(
        self,
        asset: PublishedAsset,
        *,
        owner: Caller,
        job_id: JobId | None,
        voice_version_id: VoiceVersionId | None,
        created_at: datetime,
    ) -> None:
        """Record one published asset; re-recording the same location after a retry is a no-op."""
        location = asset.stored.location
        bucket = location.uri.removeprefix("s3://").split("/", 1)[0] if location.backend is StorageBackend.S3 else None
        statement = (
            insert(MediaAssetRow)
            .values(
                id=new_asset_id(),
                job_id=job_id,
                voice_version_id=voice_version_id,
                owner_issuer=owner.issuer,
                owner_subject=owner.subject,
                role=asset.role.value,
                name=asset.name,
                storage_backend=location.backend.value,
                uri=location.uri,
                bucket=bucket,
                object_key=asset.key.value,
                sha256=asset.stored.sha256,
                size_bytes=asset.stored.size_bytes,
                content_type=asset.content_type,
                duration_seconds=asset.duration_seconds,
                sample_rate=asset.sample_rate,
                created_at=created_at,
            )
            .on_conflict_do_nothing(index_elements=["uri"])
        )
        await self._session.execute(statement)

    async def locations_for_voice(self, voice_id: VoiceId) -> list[AssetLocation]:
        """Return every stored recording of every version of a voice that belongs to the voice's owner."""
        rows = await self._session.execute(
            select(MediaAssetRow.storage_backend, MediaAssetRow.uri)
            .where(MediaAssetRow.id.in_(_voice_asset_ids(voice_id)))
            .order_by(MediaAssetRow.uri)
        )
        return [AssetLocation(StorageBackend(backend), uri) for backend, uri in rows]

    async def locations_for_job(self, job_id: JobId) -> list[AssetLocation]:
        """Return the locations of every asset of a job."""
        rows = await self._session.execute(
            select(MediaAssetRow.storage_backend, MediaAssetRow.uri).where(MediaAssetRow.job_id == job_id)
        )
        return [AssetLocation(StorageBackend(backend), uri) for backend, uri in rows]

    async def delete_for_voice(self, voice_id: VoiceId) -> int:
        """Delete the asset rows returned by locations_for_voice; return the count."""
        result = await self._session.execute(
            delete(MediaAssetRow).where(MediaAssetRow.id.in_(_voice_asset_ids(voice_id)))
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def known_uris(self, uris: list[str]) -> set[str]:
        """Return which of the given URIs are recorded or are the uploaded input of an unfinished job."""
        if not uris:
            return set()
        input_uri = JobRow.input_payload[INPUT_LOCATION_FIELD]["uri"].astext
        recorded = select(MediaAssetRow.uri).where(MediaAssetRow.uri.in_(uris))
        pending = select(input_uri).where(input_uri.in_(uris), JobRow.status.not_in(TERMINAL_STATUSES))
        rows = await self._session.scalars(union(recorded, pending))
        return set(rows)
