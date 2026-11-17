"""In-memory STT backend."""

import threading
from pathlib import Path

from app.domain.errors import InferenceError
from app.domain.models import InstalledModel
from app.domain.stt import (
    Segment,
    SttBackendName,
    SttCapabilities,
    SttRequest,
    SttRuntime,
    TranscriptionResult,
    Word,
)

FULL_CAPABILITIES = SttCapabilities(
    beam_search=True,
    hotwords=True,
    initial_prompt=True,
    native_vad=True,
    word_timestamps=True,
    language_probability=True,
)
MLX_LIKE_CAPABILITIES = SttCapabilities(
    beam_search=False,
    hotwords=False,
    initial_prompt=True,
    native_vad=False,
    word_timestamps=True,
    language_probability=False,
)


class MockSttBackend:
    """Returns a fixed transcript and records requests."""

    def __init__(
        self,
        *,
        text: str = "hello world",
        capabilities: SttCapabilities = FULL_CAPABILITIES,
        name: SttBackendName = SttBackendName.CTRANSLATE2,
        error: Exception | None = None,
    ) -> None:
        """Configure the transcript, capabilities and an optional error to raise."""
        self._text = text
        self._capabilities = capabilities
        self._name = name
        self.error = error
        self.loaded: InstalledModel | None = None
        self.requests: list[SttRequest] = []
        self.load_count = 0

    @property
    def name(self) -> SttBackendName:
        """Return the backend identifier."""
        return self._name

    @property
    def capabilities(self) -> SttCapabilities:
        """Return the configured capabilities."""
        return self._capabilities

    @property
    def runtime(self) -> SttRuntime:
        """Return a CPU runtime."""
        return SttRuntime(backend=self._name, device="cpu", compute_type="int8")

    def load(self, model: InstalledModel) -> None:
        """Record the loaded model."""
        self.loaded = model
        self.load_count += 1

    def transcribe(self, request: SttRequest) -> TranscriptionResult:
        """Return two segments spanning the text, or raise the configured error."""
        if self.loaded is None:
            raise InferenceError("no STT model loaded")
        if self.error is not None:
            raise self.error
        self.requests.append(request)
        words = self._text.split()
        word_items = tuple(Word(text=w, start=i * 0.5, end=i * 0.5 + 0.4, probability=0.9) for i, w in enumerate(words))
        segment = Segment(
            start=0.0,
            end=len(words) * 0.5,
            text=self._text,
            words=word_items,
            no_speech_prob=0.01,
            avg_logprob=-0.2,
            compression_ratio=1.2,
        )
        return TranscriptionResult(
            segments=(segment,), language=request.language or "en", language_probability=0.99, duration_seconds=2.0
        )

    def unload(self) -> None:
        """Forget the loaded model."""
        self.loaded = None


def audio_path_of(request: SttRequest) -> Path:
    """Return the audio path a request pointed at."""
    return request.audio_path


class MockBlockingSttBackend(MockSttBackend):
    """Blocks inside transcribe on the inference thread, as a native call that cannot be cancelled does."""

    def __init__(self) -> None:
        """Create the events the test uses to observe and end the blocked call."""
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, request: SttRequest) -> TranscriptionResult:
        """Signal the start and block until the test releases the call."""
        self.started.set()
        self.release.wait()
        return super().transcribe(request)
