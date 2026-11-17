"""Runtime catalogs: live answers from the local inference components, or the snapshot a GPU worker published."""

import asyncio
import logging

from app.domain.errors import CapabilityUnavailableError
from app.domain.models import ModelKind
from app.domain.stt import SttCapabilities, SttRuntime
from app.domain.tts import TtsBackendName, TtsCapabilities
from app.domain.workers import CapabilitySnapshot, InstalledModelRef
from app.ports.models import ModelCatalog
from app.ports.unit_of_work import UnitOfWorkFactory
from app.services.model_host import ModelHost

logger = logging.getLogger(__name__)
NO_WORKER_MESSAGE = "no GPU worker has registered yet; start it once with `python -m app worker --register-only`"


class LocalRuntimeCatalog:
    """Answers from the in-process inference components; used by the all and worker roles."""

    def __init__(
        self,
        *,
        model_host: ModelHost,
        catalog: ModelCatalog,
        profile: str,
        denoise_device: str | None,
    ) -> None:
        """Store the model_host, the installed-model catalog and the selected profile and denoise device."""
        self._model_host = model_host
        self._catalog = catalog
        self._profile = profile
        self._denoise_device = denoise_device
        self._preload_errors: tuple[str, ...] = ()

    def stt_runtime(self) -> SttRuntime:
        """Return the STT backend, device and compute type."""
        return self._model_host.stt.runtime

    def stt_capabilities(self) -> SttCapabilities:
        """Return the options the STT backend honors."""
        return self._model_host.stt.capabilities

    def tts_backend(self) -> TtsBackendName | None:
        """Return the TTS runtime, or None without one."""
        tts = self._model_host.tts
        return tts.name if tts else None

    def tts_capabilities(self) -> TtsCapabilities | None:
        """Return the TTS features, or None without a TTS runtime."""
        tts = self._model_host.tts
        return tts.capabilities if tts else None

    def denoise_available(self) -> bool:
        """Return whether denoising can run."""
        return self._denoise_device is not None

    def installed_stt_names(self) -> frozenset[str]:
        """Return the STT model names installed for the active backend."""
        return self._model_host.installed_stt_names()

    def has_model(self, kind: ModelKind, backend: str, name: str) -> bool:
        """Return whether a model is installed."""
        return self._catalog.find(kind, backend, name) is not None

    def preset_speakers(self, checkpoint: str) -> tuple[str, ...]:
        """Return the preset speakers of an installed CustomVoice checkpoint."""
        return self._model_host.preset_speakers(self._model_host.tts_checkpoint(checkpoint))

    def loaded_models(self) -> dict[str, str | None]:
        """Return the names of resident models."""
        return self._model_host.loaded_models

    def record_preload_errors(self, errors: tuple[str, ...]) -> None:
        """Remember preload failures so the published snapshot reports them."""
        self._preload_errors = errors

    def snapshot(self) -> CapabilitySnapshot:
        """Return the current capabilities, installed models, preset speakers and resident models."""
        installed = self._catalog.installed()
        tts = self._model_host.tts
        speakers: dict[str, tuple[str, ...]] = {}
        if tts is not None:
            for model in installed:
                if model.kind is ModelKind.TTS and model.backend == str(tts.name):
                    names = tts.preset_speakers(model)
                    if names:
                        speakers[model.name] = names
        return CapabilitySnapshot(
            profile=self._profile,
            stt_runtime=self.stt_runtime(),
            stt_capabilities=self.stt_capabilities(),
            tts_backend=self.tts_backend(),
            tts_device=tts.device if tts else None,
            tts_capabilities=self.tts_capabilities(),
            denoise_device=self._denoise_device,
            installed=frozenset(InstalledModelRef(model.kind, model.backend, model.name) for model in installed),
            preset_speakers=speakers,
            loaded=self.loaded_models(),
            preload_errors=self._preload_errors,
        )


class RegistryRuntimeCatalog:
    """Answers from the snapshot of the active or most recent registered worker; used by the api role."""

    def __init__(self, *, uow: UnitOfWorkFactory, refresh_seconds: float) -> None:
        """Store the unit of work and the refresh interval; nothing is known until the first refresh."""
        self._uow = uow
        self._refresh = refresh_seconds
        self._snapshot: CapabilitySnapshot | None = None

    async def refresh(self) -> None:
        """Read the newest registered worker's snapshot."""
        async with self._uow() as uow:
            record = await uow.workers.latest()
        if record is not None:
            self._snapshot = record.capabilities

    async def run(self) -> None:
        """Refresh periodically until cancelled; this loop is a task boundary that logs and survives failures."""
        while True:
            try:
                await self.refresh()
            except Exception:
                logger.exception("runtime catalog refresh failed")
            await asyncio.sleep(self._refresh)

    def snapshot(self) -> CapabilitySnapshot:
        """Return the cached snapshot or raise capability_unavailable when no worker has registered."""
        if self._snapshot is None:
            raise CapabilityUnavailableError(NO_WORKER_MESSAGE)
        return self._snapshot

    def stt_runtime(self) -> SttRuntime:
        """Return the STT backend, device and compute type."""
        return self.snapshot().stt_runtime

    def stt_capabilities(self) -> SttCapabilities:
        """Return the options the STT backend honors."""
        return self.snapshot().stt_capabilities

    def tts_backend(self) -> TtsBackendName | None:
        """Return the TTS runtime, or None without one."""
        return self.snapshot().tts_backend

    def tts_capabilities(self) -> TtsCapabilities | None:
        """Return the TTS features, or None without a TTS runtime."""
        return self.snapshot().tts_capabilities

    def denoise_available(self) -> bool:
        """Return whether denoising can run."""
        return self.snapshot().denoise_available

    def installed_stt_names(self) -> frozenset[str]:
        """Return the STT model names installed for the active backend."""
        snapshot = self.snapshot()
        return snapshot.installed_names(ModelKind.STT, str(snapshot.stt_runtime.backend))

    def has_model(self, kind: ModelKind, backend: str, name: str) -> bool:
        """Return whether a model is installed."""
        return InstalledModelRef(kind, backend, name) in self.snapshot().installed

    def preset_speakers(self, checkpoint: str) -> tuple[str, ...]:
        """Return the preset speakers of an installed CustomVoice checkpoint."""
        return tuple(self.snapshot().preset_speakers.get(checkpoint, ()))

    def loaded_models(self) -> dict[str, str | None]:
        """Return the names of resident models."""
        return dict(self.snapshot().loaded)
