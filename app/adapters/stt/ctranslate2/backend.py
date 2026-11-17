"""faster-whisper on CTranslate2 (CUDA or CPU) with its built-in Silero VAD."""

import gc
from typing import Any

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from app.domain.errors import InferenceError, MediaError
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

WHISPER_SAMPLE_RATE = 16000

CAPABILITIES = SttCapabilities(
    beam_search=True,
    hotwords=True,
    initial_prompt=True,
    native_vad=True,
    word_timestamps=True,
    language_probability=True,
)


def _is_out_of_memory(exc: BaseException) -> bool:
    """Return whether an exception reports exhausted device memory."""
    text = str(exc).lower()
    return "out of memory" in text or "cuda_error_out_of_memory" in text


class CTranslate2Backend:
    """SttBackend backed by faster-whisper."""

    def __init__(self, runtime: SttRuntime, *, cpu_threads: int = 0) -> None:
        """Store the runtime; the model loads lazily."""
        self._runtime = runtime
        self._cpu_threads = cpu_threads
        self._model: WhisperModel | None = None

    @property
    def name(self) -> SttBackendName:
        """Return the backend identifier."""
        return SttBackendName.CTRANSLATE2

    @property
    def capabilities(self) -> SttCapabilities:
        """Return the options this backend can honor."""
        return CAPABILITIES

    @property
    def runtime(self) -> SttRuntime:
        """Return the device and compute type in use."""
        return self._runtime

    def load(self, model: InstalledModel) -> None:
        """Load an installed model from its local directory without network access."""
        self.unload()
        try:
            self._model = WhisperModel(
                str(model.local_path),
                device=self._runtime.device,
                compute_type=self._runtime.compute_type,
                cpu_threads=self._cpu_threads,
                local_files_only=True,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise InferenceError(
                f"could not load STT model {model.name}", retryable=not _is_out_of_memory(exc)
            ) from exc

    def transcribe(self, request: SttRequest) -> TranscriptionResult:
        """Transcribe one prepared audio file."""
        model = self._require_model()
        vad = request.vad
        samples, sample_rate = sf.read(str(request.audio_path), dtype="float32", always_2d=False)
        if sample_rate != WHISPER_SAMPLE_RATE or np.ndim(samples) != 1:
            raise MediaError("STT input must be 16 kHz mono audio")
        try:
            segments, info = model.transcribe(
                np.ascontiguousarray(samples, dtype=np.float32),
                language=request.language,
                beam_size=request.beam_size,
                temperature=request.temperature,
                word_timestamps=request.word_timestamps,
                initial_prompt=request.initial_prompt,
                hotwords=" ".join(request.hotwords) or None,
                condition_on_previous_text=request.condition_on_previous_text,
                no_speech_threshold=request.no_speech_threshold,
                vad_filter=vad.enabled,
                vad_parameters={
                    "threshold": vad.threshold,
                    "min_speech_duration_ms": vad.min_speech_duration_ms,
                    "min_silence_duration_ms": vad.min_silence_duration_ms,
                    "speech_pad_ms": vad.speech_pad_ms,
                },
            )
            mapped = tuple(self._segment(segment) for segment in segments)
        except (RuntimeError, ValueError) as exc:
            raise InferenceError("STT inference failed", retryable=not _is_out_of_memory(exc)) from exc
        return TranscriptionResult(
            segments=mapped,
            language=info.language,
            language_probability=info.language_probability,
            duration_seconds=info.duration,
        )

    def unload(self) -> None:
        """Release the loaded model."""
        if self._model is not None:
            self._model = None
            gc.collect()

    def _require_model(self) -> WhisperModel:
        """Return the loaded model or raise InferenceError."""
        if self._model is None:
            raise InferenceError("no STT model loaded")
        return self._model

    @staticmethod
    def _segment(segment: Any) -> Segment:
        """Map a faster-whisper segment to the domain type."""
        words = tuple(
            Word(text=word.word.strip(), start=word.start, end=word.end, probability=word.probability)
            for word in (segment.words or ())
        )
        return Segment(
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
            words=words,
            no_speech_prob=segment.no_speech_prob,
            avg_logprob=segment.avg_logprob,
            compression_ratio=segment.compression_ratio,
        )
