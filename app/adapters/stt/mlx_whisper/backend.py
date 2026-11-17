"""mlx-whisper on Apple Silicon; VAD runs as a separate Silero stage before this backend."""

import gc
from typing import Any

import mlx.core as mx
import mlx_whisper
from mlx_whisper.transcribe import ModelHolder

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

CAPABILITIES = SttCapabilities(
    beam_search=False,
    hotwords=False,
    initial_prompt=True,
    native_vad=False,
    word_timestamps=True,
    language_probability=False,
)
RUNTIME = SttRuntime(backend=SttBackendName.MLX, device="metal", compute_type="float16")


def _optional_float(value: Any) -> float | None:
    """Return value as float, or None when absent."""
    return float(value) if value is not None else None


class MlxWhisperBackend:
    """SttBackend backed by mlx-whisper with a local model directory."""

    def __init__(self) -> None:
        """Start unloaded."""
        self._model_path: str | None = None

    @property
    def name(self) -> SttBackendName:
        """Return the backend identifier."""
        return SttBackendName.MLX

    @property
    def capabilities(self) -> SttCapabilities:
        """Return mlx-whisper features: greedy decoding, no hotwords, no built-in VAD."""
        return CAPABILITIES

    @property
    def runtime(self) -> SttRuntime:
        """Return the Metal float16 runtime."""
        return RUNTIME

    def load(self, model: InstalledModel) -> None:
        """Load local MLX weights into mlx-whisper's model holder."""
        path = str(model.local_path)
        try:
            ModelHolder.get_model(path, mx.float16)
        except (RuntimeError, ValueError, OSError) as exc:
            raise InferenceError(f"could not load STT model {model.name}") from exc
        self._model_path = path

    def transcribe(self, request: SttRequest) -> TranscriptionResult:
        """Transcribe one prepared audio file with greedy decoding."""
        if self._model_path is None:
            raise InferenceError("no STT model loaded")
        try:
            output = mlx_whisper.transcribe(
                str(request.audio_path),
                path_or_hf_repo=self._model_path,
                language=request.language,
                temperature=request.temperature,
                word_timestamps=request.word_timestamps,
                initial_prompt=request.initial_prompt,
                condition_on_previous_text=request.condition_on_previous_text,
                no_speech_threshold=request.no_speech_threshold,
                verbose=None,
            )
        except (RuntimeError, ValueError) as exc:
            raise InferenceError("STT inference failed", retryable=True) from exc
        segments = tuple(self._segment(item) for item in output.get("segments", ()))
        duration = segments[-1].end if segments else 0.0
        return TranscriptionResult(
            segments=segments,
            language=str(output.get("language") or request.language or ""),
            language_probability=None,
            duration_seconds=duration,
        )

    def unload(self) -> None:
        """Drop the cached model so unified memory can be reclaimed."""
        ModelHolder.model = None
        ModelHolder.model_path = None
        self._model_path = None
        gc.collect()

    @staticmethod
    def _segment(item: dict[str, Any]) -> Segment:
        """Map an mlx-whisper segment dictionary to the domain type."""
        words = tuple(
            Word(
                text=str(word["word"]).strip(),
                start=float(word["start"]),
                end=float(word["end"]),
                probability=_optional_float(word.get("probability")),
            )
            for word in item.get("words") or ()
        )
        return Segment(
            start=float(item["start"]),
            end=float(item["end"]),
            text=str(item["text"]).strip(),
            words=words,
            no_speech_prob=_optional_float(item.get("no_speech_prob")),
            avg_logprob=_optional_float(item.get("avg_logprob")),
            compression_ratio=_optional_float(item.get("compression_ratio")),
        )
