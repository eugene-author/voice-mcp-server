"""Stand-in for the silero_vad module."""

import sys
import types
from typing import Any

import pytest


class MockSileroVad:
    """Records VAD calls and returns configured timestamps in seconds."""

    def __init__(self) -> None:
        """Start with one speech region."""
        self.loaded_with: dict[str, Any] | None = None
        self.calls: list[dict[str, Any]] = []
        self.timestamps: list[dict[str, float]] = [{"start": 0.25, "end": 1.0}]

    def load_silero_vad(self, **kwargs: Any) -> object:
        """Record load arguments and return a model token."""
        self.loaded_with = kwargs
        return object()

    def get_speech_timestamps(self, audio: Any, model: Any, **kwargs: Any) -> list[dict[str, float]]:
        """Record the call and return the configured timestamps."""
        self.calls.append({"length": len(audio), **kwargs})
        return self.timestamps


def install(monkeypatch: pytest.MonkeyPatch) -> MockSileroVad:
    """Install a mock silero_vad module and return the recorder."""
    recorder = MockSileroVad()
    module = types.ModuleType("silero_vad")
    module.load_silero_vad = recorder.load_silero_vad  # type: ignore[attr-defined]
    module.get_speech_timestamps = recorder.get_speech_timestamps  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "silero_vad", module)
    monkeypatch.delitem(sys.modules, "app.adapters.vad.silero.detector", raising=False)
    return recorder
