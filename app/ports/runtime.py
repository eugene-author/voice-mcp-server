"""Runtime catalog port: what the inference runtime can do, answered without loading models."""

from typing import Protocol

from app.domain.models import ModelKind
from app.domain.stt import SttCapabilities, SttRuntime
from app.domain.tts import TtsBackendName, TtsCapabilities
from app.domain.workers import CapabilitySnapshot


class RuntimeCatalog(Protocol):
    """Capabilities and installed models of the runtime that executes jobs; raises capability_unavailable if unknown."""

    def stt_runtime(self) -> SttRuntime:
        """Return the STT backend, device and compute type."""
        ...

    def stt_capabilities(self) -> SttCapabilities:
        """Return the options the STT backend honors."""
        ...

    def tts_backend(self) -> TtsBackendName | None:
        """Return the TTS runtime, or None without one."""
        ...

    def tts_capabilities(self) -> TtsCapabilities | None:
        """Return the TTS features, or None without a TTS runtime."""
        ...

    def denoise_available(self) -> bool:
        """Return whether denoising can run."""
        ...

    def installed_stt_names(self) -> frozenset[str]:
        """Return the STT model names installed for the active backend."""
        ...

    def has_model(self, kind: ModelKind, backend: str, name: str) -> bool:
        """Return whether a model is installed."""
        ...

    def preset_speakers(self, checkpoint: str) -> tuple[str, ...]:
        """Return the preset speakers of an installed CustomVoice checkpoint."""
        ...

    def loaded_models(self) -> dict[str, str | None]:
        """Return the names of resident models."""
        ...

    def snapshot(self) -> CapabilitySnapshot:
        """Return everything above as one value."""
        ...
