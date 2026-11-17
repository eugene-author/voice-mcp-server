"""SileroDetector argument mapping with a mock silero_vad module."""

import importlib

import numpy as np
import pytest

from app.domain.stt import VadSettings
from app.domain.timeline import SpeechRegion
from tests.mocks import silero_vad as mock_silero


def test_loads_packaged_onnx_weights_and_maps_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the ONNX model from the package and pass the configured parameters."""
    recorder = mock_silero.install(monkeypatch)
    detector = importlib.import_module("app.adapters.vad.silero.detector").SileroDetector()
    settings = VadSettings(threshold=0.4, min_speech_duration_ms=100, min_silence_duration_ms=200, speech_pad_ms=50)
    regions = detector.detect(np.zeros(16000, dtype=np.float32), 16000, settings)
    detector.detect(np.zeros(16000, dtype=np.float32), 16000, settings)
    assert recorder.loaded_with == {"onnx": True}
    assert recorder.calls[0]["threshold"] == 0.4
    assert recorder.calls[0]["return_seconds"] is True
    assert regions == (SpeechRegion(0.25, 1.0),)


def test_rejects_wrong_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require 16 kHz audio."""
    mock_silero.install(monkeypatch)
    detector = importlib.import_module("app.adapters.vad.silero.detector").SileroDetector()
    with pytest.raises(ValueError, match="16 kHz"):
        detector.detect(np.zeros(100, dtype=np.float32), 8000, VadSettings())
