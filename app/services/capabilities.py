"""Tool availability derived from the runtime that executes jobs, local or published by a GPU worker."""

from dataclasses import dataclass

from app.domain.errors import CapabilityUnavailableError
from app.domain.jobs import (
    BatchTranscribeCommand,
    JobCommand,
    RegisterVoiceCommand,
    SynthesizeCommand,
    TranscribeCommand,
)
from app.domain.stt import SttCapabilities
from app.domain.tts import TtsCapabilities
from app.domain.workers import CapabilitySnapshot
from app.ports.runtime import RuntimeCatalog

TTS_EXTRA_HINT = "install the 'tts' extra (Linux) or the 'mlx' extra (macOS)"
DENOISE_EXTRA_HINT = "install the 'denoise' extra"


@dataclass(frozen=True, slots=True)
class HostCapabilities:
    """Which features this process can serve."""

    stt: SttCapabilities
    tts: TtsCapabilities | None
    """None when no TTS runtime extra is installed."""

    denoise: bool

    @classmethod
    def from_snapshot(cls, snapshot: CapabilitySnapshot) -> HostCapabilities:
        """Derive capabilities from a worker's capability snapshot."""
        return cls(stt=snapshot.stt_capabilities, tts=snapshot.tts_capabilities, denoise=snapshot.denoise_available)

    @classmethod
    def from_runtime(cls, runtime: RuntimeCatalog) -> HostCapabilities:
        """Derive capabilities from a runtime catalog, without building its full snapshot."""
        return cls(stt=runtime.stt_capabilities(), tts=runtime.tts_capabilities(), denoise=runtime.denoise_available())

    def require_tts(self, feature: str) -> TtsCapabilities:
        """Return TTS capabilities or raise capability_unavailable naming the missing extra."""
        if self.tts is None:
            raise CapabilityUnavailableError(f"{feature} requires a TTS runtime; {TTS_EXTRA_HINT}")
        return self.tts

    def require_denoise(self) -> None:
        """Raise capability_unavailable when denoising was explicitly requested but is not installed."""
        if not self.denoise:
            raise CapabilityUnavailableError(f"denoising requires DeepFilterNet; {DENOISE_EXTRA_HINT}")

    def require_for(self, command: JobCommand) -> None:
        """Raise capability_unavailable if the command needs a component this host lacks."""
        match command:
            case SynthesizeCommand():
                self.require_tts("text_to_speech")
            case RegisterVoiceCommand():
                self.require_tts("register_voice")
            case TranscribeCommand(options=options) if options is not None and options.denoise:
                self.require_denoise()
            case TranscribeCommand() | BatchTranscribeCommand():
                return
