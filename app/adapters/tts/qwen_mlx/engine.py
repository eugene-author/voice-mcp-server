"""Qwen3-TTS through mlx-audio on Apple Silicon."""

import gc
import json
from typing import Any

import numpy as np
from mlx_audio.tts.utils import load_model

from app.domain.errors import InferenceError, InvalidRequestError
from app.domain.models import InstalledModel, TtsVariant
from app.domain.tts import (
    CloneReference,
    PresetSpeaker,
    SynthesisRequest,
    SynthesizedAudio,
    TtsBackendName,
    TtsCapabilities,
)

CAPABILITIES = TtsCapabilities(
    clone_icl=True,
    clone_xvector_only=False,
    preset_voices=True,
    reusable_clone_prompt=False,
)
DEFAULT_SAMPLE_RATE = 24000


class MlxQwen3TtsEngine:
    """TtsEngine backed by mlx-audio; conditioning is rebuilt from the reference on every call."""

    def __init__(self) -> None:
        """Start unloaded; the untyped mlx-audio model stays behind this boundary."""
        self._model: Any | None = None
        self._checkpoint: InstalledModel | None = None

    @property
    def name(self) -> TtsBackendName:
        """Return the runtime identifier."""
        return TtsBackendName.MLX

    @property
    def capabilities(self) -> TtsCapabilities:
        """Return mlx-audio features: ICL cloning only, no reusable prompt."""
        return CAPABILITIES

    @property
    def device(self) -> str:
        """Return the Metal device name."""
        return "metal"

    def load(self, checkpoint: InstalledModel) -> None:
        """Load local MLX weights."""
        try:
            self._model = load_model(checkpoint.local_path)
        except (RuntimeError, ValueError, OSError) as exc:
            raise InferenceError(f"could not load TTS checkpoint {checkpoint.name}") from exc
        self._checkpoint = checkpoint

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        """Synthesize one chunk with a cloned or preset voice."""
        model, checkpoint = self._require_loaded()
        try:
            match request.voice:
                case CloneReference(audio_path=path, text=str() as ref_text):
                    if checkpoint.variant is not TtsVariant.BASE:
                        raise InvalidRequestError("cloned voices require a Base checkpoint")
                    results = model.generate(
                        text=request.text, lang_code=request.language, ref_audio=str(path), ref_text=ref_text
                    )
                case CloneReference():
                    raise InvalidRequestError("mlx TTS supports ICL cloning only; reference_text is required")
                case PresetSpeaker(name=speaker):
                    if checkpoint.variant is not TtsVariant.CUSTOM_VOICE:
                        raise InvalidRequestError("preset speakers require a CustomVoice checkpoint")
                    results = model.generate_custom_voice(text=request.text, speaker=speaker, language=request.language)
            chunks = [np.asarray(result.audio, dtype=np.float32).reshape(-1) for result in results]
        except (RuntimeError, ValueError) as exc:
            raise InferenceError("TTS inference failed", retryable=True) from exc
        if not chunks:
            raise InferenceError("mlx TTS returned no audio")
        return SynthesizedAudio(samples=np.concatenate(chunks), sample_rate=self._sample_rate(model))

    def preset_speakers(self, checkpoint: InstalledModel) -> tuple[str, ...]:
        """Return speaker names from the checkpoint's config.json without loading weights."""
        if checkpoint.variant is not TtsVariant.CUSTOM_VOICE:
            return ()
        config = json.loads((checkpoint.local_path / "config.json").read_text(encoding="utf-8"))
        speakers = (config.get("talker_config") or {}).get("spk_id") or {}
        return tuple(sorted(str(name) for name in speakers))

    def unload(self) -> None:
        """Drop the model so its unified memory can be reclaimed."""
        self._model = None
        self._checkpoint = None
        gc.collect()

    def _require_loaded(self) -> tuple[Any, InstalledModel]:
        """Return the loaded model and checkpoint or raise InferenceError."""
        if self._model is None or self._checkpoint is None:
            raise InferenceError("no TTS checkpoint loaded")
        return self._model, self._checkpoint

    @staticmethod
    def _sample_rate(model: Any) -> int:
        """Return the output sample rate exposed by the pinned mlx-audio model."""
        return int(getattr(model, "sample_rate", DEFAULT_SAMPLE_RATE))
