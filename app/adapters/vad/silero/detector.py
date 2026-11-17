"""Standalone Silero VAD from the installed silero-vad package weights (ONNX), not torch.hub."""

import numpy as np
import numpy.typing as npt
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from app.domain.stt import VadSettings
from app.domain.timeline import SpeechRegion

SILERO_SAMPLE_RATE = 16000


class SileroDetector:
    """VoiceActivityDetector for the mlx STT path."""

    def __init__(self) -> None:
        """Start without a loaded model; it loads on first use from package data."""
        self._model: object | None = None

    def detect(
        self, samples: npt.NDArray[np.float32], sample_rate: int, settings: VadSettings
    ) -> tuple[SpeechRegion, ...]:
        """Return speech regions in seconds for 16 kHz mono samples."""
        if sample_rate != SILERO_SAMPLE_RATE:
            raise ValueError("Silero VAD expects 16 kHz audio")
        if self._model is None:
            self._model = load_silero_vad(onnx=True)
        timestamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)),
            self._model,
            threshold=settings.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=settings.min_speech_duration_ms,
            min_silence_duration_ms=settings.min_silence_duration_ms,
            speech_pad_ms=settings.speech_pad_ms,
            return_seconds=True,
        )
        return tuple(SpeechRegion(start=float(item["start"]), end=float(item["end"])) for item in timestamps)
