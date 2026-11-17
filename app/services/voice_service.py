"""Voices: listing, deletion and execution of registration jobs."""

import logging

from app.domain.caller import Caller
from app.domain.errors import InvalidRequestError, NotFoundError
from app.domain.ids import VoiceId
from app.domain.models import ModelKind
from app.domain.voices import PRESET_VOICE_PREFIX, VoiceKind, VoiceStatus, VoiceSummary
from app.ports.clock import Clock
from app.ports.runtime import RuntimeCatalog
from app.ports.storage import AssetStore
from app.ports.unit_of_work import UnitOfWorkFactory
from app.services.capabilities import HostCapabilities

logger = logging.getLogger(__name__)


class VoiceService:
    """Owner-scoped voice queries and authorized deletion."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkFactory,
        store: AssetStore,
        runtime: RuntimeCatalog,
        preset_checkpoint: str | None,
        clock: Clock,
    ) -> None:
        """Store injected collaborators."""
        self._uow = uow
        self._store = store
        self._runtime = runtime
        self._preset_checkpoint = preset_checkpoint
        self._clock = clock

    async def list_voices(self, *, owner: Caller, language: str | None) -> list[VoiceSummary]:
        """Return preset speakers of the installed CustomVoice checkpoint and the owner's cloned voices."""
        HostCapabilities.from_runtime(self._runtime).require_tts("list_voices")
        summaries = self._preset_summaries()
        async with self._uow() as uow:
            voices = await uow.voices.list_owned(owner, language=language.lower() if language else None)
        summaries.extend(
            VoiceSummary(
                voice_id=voice.id, name=voice.name, kind=VoiceKind.CLONED, status=voice.status, language=voice.language
            )
            for voice in voices
        )
        return summaries

    async def delete_voice(self, voice_id: VoiceId, *, owner: Caller | None) -> int:
        """Mark the voice deleted, purge stored recordings of all its versions, then their rows; return the count."""
        async with self._uow() as uow:
            voice = await uow.voices.get(voice_id, owner, include_deleted=True)
            if voice is None:
                raise NotFoundError("voice not found")
            if not await uow.voices.mark_deleted(voice_id, now=self._clock.now()):
                raise InvalidRequestError(
                    "the voice registration is still in progress; cancel its job or wait for it to finish"
                )
            locations = await uow.assets.locations_for_voice(voice_id)
        for location in locations:
            await self._store.delete(location)
        async with self._uow() as uow:
            await uow.assets.delete_for_voice(voice_id)
        logger.info("voice deleted", extra={"voice_id": voice_id, "assets": len(locations)})
        return len(locations)

    def _preset_summaries(self) -> list[VoiceSummary]:
        """Return preset speakers when the CustomVoice checkpoint is installed for the active runtime."""
        capabilities, backend = self._runtime.tts_capabilities(), self._runtime.tts_backend()
        if capabilities is None or backend is None or not capabilities.preset_voices or not self._preset_checkpoint:
            return []
        if not self._runtime.has_model(ModelKind.TTS, str(backend), self._preset_checkpoint):
            return []
        return [
            VoiceSummary(
                voice_id=f"{PRESET_VOICE_PREFIX}{speaker}",
                name=speaker,
                kind=VoiceKind.PRESET,
                status=VoiceStatus.READY,
            )
            for speaker in self._runtime.preset_speakers(self._preset_checkpoint)
        ]
