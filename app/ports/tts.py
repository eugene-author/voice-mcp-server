"""Text-to-speech engine port."""

from typing import Protocol

from app.domain.models import InstalledModel
from app.domain.tts import SynthesisRequest, SynthesizedAudio, TtsBackendName, TtsCapabilities


class TtsEngine(Protocol):
    """Blocking TTS interface, always called on the inference executor thread."""

    @property
    def name(self) -> TtsBackendName:
        """Return the runtime identifier."""
        ...

    @property
    def capabilities(self) -> TtsCapabilities:
        """Return the features this runtime supports."""
        ...

    @property
    def device(self) -> str:
        """Return the device the runtime runs on."""
        ...

    def load(self, checkpoint: InstalledModel) -> None:
        """Load an installed checkpoint from its local directory."""
        ...

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        """Synthesize one text chunk."""
        ...

    def preset_speakers(self, checkpoint: InstalledModel) -> tuple[str, ...]:
        """Return a checkpoint's preset speakers from its metadata without loading weights."""
        ...

    def unload(self) -> None:
        """Release the loaded checkpoint."""
        ...
