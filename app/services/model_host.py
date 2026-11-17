"""Lazy loading and eviction of STT, TTS and denoise models, all on the inference thread."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from app.core.executor import InferenceRunner
from app.domain.errors import CapabilityUnavailableError, ModelNotInstalledError
from app.domain.models import InstalledModel, ModelKind
from app.domain.stt import SttRequest, TranscriptionResult, VadSettings
from app.domain.timeline import SpeechRegion
from app.domain.tts import SynthesisRequest, SynthesizedAudio
from app.domain.workers import ModelRef
from app.ports.denoise import Denoiser
from app.ports.models import ModelCatalog
from app.ports.stt import SttBackend
from app.ports.tts import TtsEngine
from app.ports.vad import VoiceActivityDetector
from app.services.capabilities import DENOISE_EXTRA_HINT, TTS_EXTRA_HINT

logger = logging.getLogger(__name__)


class ModelHost:
    """Keeps at most one STT and one TTS model resident, or only one of them on small hosts."""

    def __init__(
        self,
        *,
        stt: SttBackend,
        tts: TtsEngine | None,
        denoiser: Denoiser | None,
        vad: VoiceActivityDetector | None,
        catalog: ModelCatalog,
        executor: InferenceRunner,
        exclusive: bool,
        denoise_model: str = "deepfilternet3",
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Store components; models load lazily on first use, or up front through preload."""
        self._stt = stt
        self._tts = tts
        self._denoiser = denoiser
        self._vad = vad
        self._catalog = catalog
        self._executor = executor
        self._exclusive = exclusive
        self._denoise_model = denoise_model
        self._stt_loaded: InstalledModel | None = None
        self._tts_loaded: InstalledModel | None = None
        self._denoise_loaded = False
        self._monotonic = monotonic
        self._last_used: dict[ModelKind, float] = {}

    @property
    def stt(self) -> SttBackend:
        """Return the STT backend."""
        return self._stt

    @property
    def tts(self) -> TtsEngine | None:
        """Return the TTS engine, if installed."""
        return self._tts

    @property
    def loaded_models(self) -> dict[str, str | None]:
        """Return the names of currently resident models."""
        return {
            "stt": self._stt_loaded.name if self._stt_loaded else None,
            "tts": self._tts_loaded.name if self._tts_loaded else None,
        }

    def stt_model(self, name: str) -> InstalledModel:
        """Return an installed STT model for the active backend or raise model_not_installed."""
        return self._require(ModelKind.STT, str(self._stt.name), name)

    def installed_stt_names(self) -> frozenset[str]:
        """Return the STT model names installed for the active backend."""
        backend = str(self._stt.name)
        return frozenset(model.name for model in self._catalog.installed(ModelKind.STT) if model.backend == backend)

    def tts_checkpoint(self, name: str) -> InstalledModel:
        """Return an installed TTS checkpoint for the active runtime or raise."""
        tts = self._require_tts()
        return self._require(ModelKind.TTS, str(tts.name), name)

    async def transcribe(self, model: InstalledModel, request: SttRequest) -> TranscriptionResult:
        """Load the model if needed and transcribe on the inference thread."""
        return await self._executor.run(self._transcribe_sync, model, request)

    async def synthesize(self, checkpoint: InstalledModel, request: SynthesisRequest) -> SynthesizedAudio:
        """Load the checkpoint if needed and synthesize one chunk on the inference thread."""
        return await self._executor.run(self._synthesize_sync, checkpoint, request)

    def preset_speakers(self, checkpoint: InstalledModel) -> tuple[str, ...]:
        """Return the preset speakers of a CustomVoice checkpoint from its metadata."""
        return self._require_tts().preset_speakers(checkpoint)

    async def denoise(self, source: Path, destination: Path) -> None:
        """Denoise a WAV file on the inference thread."""
        if self._denoiser is None:
            raise CapabilityUnavailableError(f"denoising requires DeepFilterNet; {DENOISE_EXTRA_HINT}")
        model = self._require(ModelKind.DENOISE, "deepfilternet", self._denoise_model)
        await self._executor.run(self._denoise_sync, model, source, destination)

    async def detect_speech(
        self, samples: npt.NDArray[np.float32], sample_rate: int, settings: VadSettings
    ) -> tuple[SpeechRegion, ...]:
        """Run Silero VAD on the inference thread."""
        if self._vad is None:
            raise CapabilityUnavailableError("standalone VAD requires the 'mlx' extra (silero-vad)")
        return await self._executor.run(self._vad.detect, samples, sample_rate, settings)

    async def unload_all(self) -> None:
        """Release every resident model."""
        await self._executor.run(self._unload_all_sync)

    async def preload(self, ref: ModelRef) -> None:
        """Load one installed model on the inference thread without running it; raise when it cannot load."""
        match ref.kind:
            case ModelKind.STT:
                await self._executor.run(self._ensure_stt_sync, self.stt_model(ref.name))
            case ModelKind.TTS:
                await self._executor.run(self._ensure_tts_sync, self.tts_checkpoint(ref.name))
            case ModelKind.DENOISE:
                if self._denoiser is None:
                    raise CapabilityUnavailableError(f"denoising requires DeepFilterNet; {DENOISE_EXTRA_HINT}")
                model = self._require(ModelKind.DENOISE, "deepfilternet", ref.name)
                await self._executor.run(self._ensure_denoiser_sync, model)

    async def unload_idle(self, *, idle_seconds: float) -> list[ModelKind]:
        """Release models unused for idle_seconds, on the inference thread so no running job loses its model."""
        return await self._executor.run(self._unload_idle_sync, idle_seconds)

    def _require(self, kind: ModelKind, backend: str, name: str) -> InstalledModel:
        """Return an installed model or raise model_not_installed."""
        model = self._catalog.find(kind, backend, name)
        if model is None:
            raise ModelNotInstalledError(
                f"{kind} model {name!r} is not installed for {backend}; run: models install --{kind} {name}"
            )
        return model

    def _require_tts(self) -> TtsEngine:
        """Return the TTS engine or raise capability_unavailable."""
        if self._tts is None:
            raise CapabilityUnavailableError(f"text_to_speech requires a TTS runtime; {TTS_EXTRA_HINT}")
        return self._tts

    def _transcribe_sync(self, model: InstalledModel, request: SttRequest) -> TranscriptionResult:
        """Ensure the STT model is resident and transcribe; runs on the inference thread."""
        self._ensure_stt_sync(model)
        try:
            return self._stt.transcribe(request)
        finally:
            self._touch(ModelKind.STT)

    def _ensure_stt_sync(self, model: InstalledModel) -> None:
        """Load an STT model, evicting the idle TTS checkpoint first when only one model may stay loaded."""
        if self._stt_loaded != model:
            if self._exclusive:
                self._unload_tts_sync()
            logger.info("loading STT model", extra={"model": model.name, "backend": str(self._stt.name)})
            self._stt.load(model)
            self._stt_loaded = model
        self._touch(ModelKind.STT)

    def _synthesize_sync(self, checkpoint: InstalledModel, request: SynthesisRequest) -> SynthesizedAudio:
        """Ensure the TTS checkpoint is resident and synthesize; runs on the inference thread."""
        self._ensure_tts_sync(checkpoint)
        try:
            return self._require_tts().synthesize(request)
        finally:
            self._touch(ModelKind.TTS)

    def _ensure_tts_sync(self, checkpoint: InstalledModel) -> None:
        """Load a TTS checkpoint, evicting the idle model first when only one model may stay loaded."""
        tts = self._require_tts()
        self._touch(ModelKind.TTS)
        if self._tts_loaded == checkpoint:
            return
        if self._exclusive:
            self._unload_stt_sync()
        logger.info("loading TTS checkpoint", extra={"model": checkpoint.name, "backend": str(tts.name)})
        tts.load(checkpoint)
        self._tts_loaded = checkpoint

    def _denoise_sync(self, model: InstalledModel, source: Path, destination: Path) -> None:
        """Load the denoiser, evicting idle STT/TTS models first when only one model may stay loaded, and run it."""
        denoiser = self._ensure_denoiser_sync(model)
        try:
            denoiser.denoise(source, destination)
        finally:
            self._touch(ModelKind.DENOISE)

    def _ensure_denoiser_sync(self, model: InstalledModel) -> Denoiser:
        """Load the denoiser, evicting idle STT/TTS models first when only one model may stay loaded."""
        if self._denoiser is None:
            raise CapabilityUnavailableError(f"denoising requires DeepFilterNet; {DENOISE_EXTRA_HINT}")
        if not self._denoise_loaded:
            if self._exclusive:
                self._unload_stt_sync()
                self._unload_tts_sync()
            self._denoiser.load(model)
            self._denoise_loaded = True
        self._touch(ModelKind.DENOISE)
        return self._denoiser

    def _touch(self, kind: ModelKind) -> None:
        """Record that a model of this kind was just used."""
        self._last_used[kind] = self._monotonic()

    def _unload_idle_sync(self, idle_seconds: float) -> list[ModelKind]:
        """Release every resident model whose last use is older than idle_seconds; runs on the inference thread."""
        cutoff = self._monotonic() - idle_seconds
        resident = {
            ModelKind.STT: self._stt_loaded is not None,
            ModelKind.TTS: self._tts_loaded is not None,
            ModelKind.DENOISE: self._denoise_loaded,
        }
        idle = [kind for kind, loaded in resident.items() if loaded and self._last_used.get(kind, 0.0) <= cutoff]
        for kind in idle:
            logger.info("unloading idle model", extra={"kind": str(kind)})
            match kind:
                case ModelKind.STT:
                    self._unload_stt_sync()
                case ModelKind.TTS:
                    self._unload_tts_sync()
                case ModelKind.DENOISE:
                    self._unload_denoiser_sync()
        return idle

    def _unload_stt_sync(self) -> None:
        """Release the STT model."""
        if self._stt_loaded is not None:
            self._stt.unload()
            self._stt_loaded = None

    def _unload_tts_sync(self) -> None:
        """Release the TTS checkpoint."""
        if self._tts is not None and self._tts_loaded is not None:
            self._tts.unload()
            self._tts_loaded = None

    def _unload_all_sync(self) -> None:
        """Release every model; runs on the inference thread."""
        self._unload_stt_sync()
        self._unload_tts_sync()
        self._unload_denoiser_sync()

    def _unload_denoiser_sync(self) -> None:
        """Release the denoiser."""
        if self._denoiser is not None and self._denoise_loaded:
            self._denoiser.unload()
            self._denoise_loaded = False
