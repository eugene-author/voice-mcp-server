"""Speech-to-text backend port."""

from typing import Protocol

from app.domain.models import InstalledModel
from app.domain.stt import SttBackendName, SttCapabilities, SttRequest, SttRuntime, TranscriptionResult


class SttBackend(Protocol):
    """Blocking STT interface, always called on the inference executor thread."""

    @property
    def name(self) -> SttBackendName:
        """Return the backend identifier."""
        ...

    @property
    def capabilities(self) -> SttCapabilities:
        """Return the options this backend can honor."""
        ...

    @property
    def runtime(self) -> SttRuntime:
        """Return the device and compute type in use."""
        ...

    def load(self, model: InstalledModel) -> None:
        """Load an installed model from its local directory."""
        ...

    def transcribe(self, request: SttRequest) -> TranscriptionResult:
        """Transcribe one prepared audio file."""
        ...

    def unload(self) -> None:
        """Release the loaded model."""
        ...
