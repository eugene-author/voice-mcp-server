"""SQLAlchemy voice repository."""

from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import VoiceRow, VoiceVersionRow
from app.domain.assets import AssetLocation
from app.domain.caller import Caller
from app.domain.ids import VoiceId, VoiceVersionId
from app.domain.tts import ConditioningMode
from app.domain.voices import NewVoice, Voice, VoiceStatus, VoiceVersion


def _version(row: VoiceVersionRow) -> VoiceVersion:
    """Map a version row to the domain type."""
    return VoiceVersion(
        id=VoiceVersionId(row.id),
        voice_id=VoiceId(row.voice_id),
        version=row.version,
        reference_uri=row.reference_uri,
        reference_backend=row.reference_backend,
        reference_sha256=row.reference_sha256,
        reference_text=row.reference_text,
        language=row.language,
        conditioning_mode=ConditioningMode(row.conditioning_mode),
    )


class SqlVoiceRepository:
    """VoiceRepository over voices and voice_versions."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to a session owned by the unit of work."""
        self._session = session

    async def add(self, voice: NewVoice) -> None:
        """Insert a pending voice with its first version."""
        self._session.add(
            VoiceRow(
                id=voice.voice_id,
                owner_issuer=voice.owner.issuer,
                owner_subject=voice.owner.subject,
                owner_client_id=voice.owner.client_id,
                name=voice.name,
                status=VoiceStatus.PENDING.value,
                language=voice.language,
                consent_confirmed=voice.consent.confirmed,
                consent_at=voice.consent.confirmed_at,
                consent_source=voice.consent.source,
                created_at=voice.created_at,
                updated_at=voice.created_at,
            )
        )
        await self._session.flush()
        self._session.add(
            VoiceVersionRow(
                id=voice.voice_version_id,
                voice_id=voice.voice_id,
                version=1,
                reference_text=voice.reference_text,
                language=voice.language,
                conditioning_mode=voice.conditioning_mode.value,
                created_at=voice.created_at,
            )
        )
        await self._session.flush()

    async def get(self, voice_id: VoiceId, owner: Caller | None, *, include_deleted: bool = False) -> Voice | None:
        """Return a voice owned by owner; deleted voices only on request, and owner None only for trusted CLI use."""
        query = select(VoiceRow).where(VoiceRow.id == voice_id)
        if not include_deleted:
            query = query.where(VoiceRow.status != VoiceStatus.DELETED.value)
        if owner is not None:
            query = query.where(VoiceRow.owner_issuer == owner.issuer, VoiceRow.owner_subject == owner.subject)
        row = await self._session.scalar(query)
        return await self._voice(row) if row is not None else None

    async def list_owned(self, owner: Caller, *, language: str | None) -> list[Voice]:
        """Return the owner's non-deleted voices, newest first."""
        query = select(VoiceRow).where(
            VoiceRow.owner_issuer == owner.issuer,
            VoiceRow.owner_subject == owner.subject,
            VoiceRow.status != VoiceStatus.DELETED.value,
        )
        if language:
            query = query.where(VoiceRow.language == language)
        rows = await self._session.scalars(query.order_by(VoiceRow.created_at.desc()))
        return [await self._voice(row) for row in rows]

    async def get_version(self, version_id: VoiceVersionId) -> VoiceVersion | None:
        """Return a voice version."""
        row = await self._session.get(VoiceVersionRow, version_id)
        return _version(row) if row is not None else None

    async def mark_ready(
        self, version_id: VoiceVersionId, *, reference: AssetLocation, sha256: str, now: datetime
    ) -> bool:
        """Move the version's voice pending to ready (voice row first), then attach the reference."""
        version_voice = select(VoiceVersionRow.voice_id).where(VoiceVersionRow.id == version_id).scalar_subquery()
        voice_id = await self._session.scalar(
            update(VoiceRow)
            .where(VoiceRow.id == version_voice, VoiceRow.status == VoiceStatus.PENDING.value)
            .values(status=VoiceStatus.READY.value, updated_at=now)
            .returning(VoiceRow.id)
        )
        if voice_id is None:
            return False
        await self._session.execute(
            update(VoiceVersionRow)
            .where(VoiceVersionRow.id == version_id)
            .values(reference_uri=reference.uri, reference_backend=reference.backend.value, reference_sha256=sha256)
        )
        return True

    async def mark_failed(self, voice_id: VoiceId, *, now: datetime) -> bool:
        """Mark a pending voice failed; return whether it changed."""
        return await self._move(voice_id, (VoiceStatus.PENDING,), VoiceStatus.FAILED, now=now)

    async def reset_to_pending(self, voice_id: VoiceId, *, now: datetime) -> bool:
        """Move a failed voice back to pending for a manually retried registration; return whether it changed."""
        return await self._move(voice_id, (VoiceStatus.FAILED,), VoiceStatus.PENDING, now=now)

    async def mark_deleted(self, voice_id: VoiceId, *, now: datetime) -> bool:
        """Mark a non-pending voice deleted (repeatable, keeps deleted_at), clear references; False if pending."""
        changed = await self._session.scalar(
            update(VoiceRow)
            .where(VoiceRow.id == voice_id, VoiceRow.status != VoiceStatus.PENDING.value)
            .values(
                status=VoiceStatus.DELETED.value, updated_at=now, deleted_at=func.coalesce(VoiceRow.deleted_at, now)
            )
            .returning(VoiceRow.id)
        )
        if changed is None:
            return False
        await self._session.execute(
            update(VoiceVersionRow)
            .where(VoiceVersionRow.voice_id == voice_id)
            .values(reference_uri=None, reference_backend=None, reference_sha256=None, reference_text=None)
        )
        return True

    async def _move(
        self, voice_id: VoiceId, sources: tuple[VoiceStatus, ...], target: VoiceStatus, *, now: datetime
    ) -> bool:
        """Change the voice status only from one of sources; return whether a row changed."""
        changed = await self._session.scalar(
            update(VoiceRow)
            .where(VoiceRow.id == voice_id, VoiceRow.status.in_([status.value for status in sources]))
            .values(status=target.value, updated_at=now)
            .returning(VoiceRow.id)
        )
        return changed is not None

    async def _voice(self, row: VoiceRow) -> Voice:
        """Map a voice row and its latest version to the domain type."""
        version_row = await self._session.scalar(
            select(VoiceVersionRow)
            .where(VoiceVersionRow.voice_id == row.id)
            .order_by(VoiceVersionRow.version.desc())
            .limit(1)
        )
        return Voice(
            id=VoiceId(row.id),
            owner=Caller(row.owner_issuer, row.owner_subject, row.owner_client_id),
            name=row.name,
            status=VoiceStatus(row.status),
            language=row.language,
            created_at=row.created_at,
            current_version=_version(version_row) if version_row is not None else None,
        )
