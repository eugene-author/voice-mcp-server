"""In-memory TTS engine."""

import numpy as np

from app.domain.errors import InferenceError
from app.domain.models import InstalledModel, TtsVariant
from app.domain.tts import (
    CloneReference,
    PresetSpeaker,
    SynthesisRequest,
    SynthesizedAudio,
    TtsBackendName,
    TtsCapabilities,
)

TORCH_LIKE = TtsCapabilities(clone_icl=True, clone_xvector_only=True, preset_voices=True, reusable_clone_prompt=True)
MLX_LIKE = TtsCapabilities(clone_icl=True, clone_xvector_only=False, preset_voices=True, reusable_clone_prompt=False)
SAMPLE_RATE = 24000


class MockTtsEngine:
    """Returns a 440 Hz tone whose length follows the text and records requests."""

    def __init__(
        self,
        *,
        capabilities: TtsCapabilities = TORCH_LIKE,
        name: TtsBackendName = TtsBackendName.TORCH,
        speakers: tuple[str, ...] = ("vivian", "ryan"),
        silent: bool = False,
    ) -> None:
        """Configure capabilities, preset speakers and whether to return silence."""
        self._capabilities = capabilities
        self._name = name
        self._speakers = speakers
        self._silent = silent
        self.loaded: InstalledModel | None = None
        self.requests: list[SynthesisRequest] = []

    @property
    def name(self) -> TtsBackendName:
        """Return the runtime identifier."""
        return self._name

    @property
    def capabilities(self) -> TtsCapabilities:
        """Return the configured capabilities."""
        return self._capabilities

    @property
    def device(self) -> str:
        """Return the CPU device."""
        return "cpu"

    def load(self, checkpoint: InstalledModel) -> None:
        """Record the loaded checkpoint."""
        self.loaded = checkpoint

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        """Return a tone for the chunk, enforcing checkpoint variants like the real runtimes."""
        if self.loaded is None:
            raise InferenceError("no TTS checkpoint loaded")
        match request.voice:
            case CloneReference() if self.loaded.variant is not TtsVariant.BASE:
                raise InferenceError("cloned voices require a Base checkpoint")
            case PresetSpeaker() if self.loaded.variant is not TtsVariant.CUSTOM_VOICE:
                raise InferenceError("preset speakers require a CustomVoice checkpoint")
        self.requests.append(request)
        length = max(SAMPLE_RATE // 10, len(request.text) * SAMPLE_RATE // 20)
        if self._silent:
            return SynthesizedAudio(samples=np.zeros(length, dtype=np.float32), sample_rate=SAMPLE_RATE)
        tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(length) / SAMPLE_RATE)
        return SynthesizedAudio(samples=tone.astype(np.float32), sample_rate=SAMPLE_RATE)

    def preset_speakers(self, checkpoint: InstalledModel) -> tuple[str, ...]:
        """Return the configured speakers for CustomVoice checkpoints."""
        return self._speakers if checkpoint.variant is TtsVariant.CUSTOM_VOICE else ()

    def unload(self) -> None:
        """Forget the loaded checkpoint."""
        self.loaded = None
