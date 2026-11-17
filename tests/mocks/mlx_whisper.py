"""Stand-ins for the mlx, mlx.core and mlx_whisper modules, installed into sys.modules on Linux."""

import sys
import types
from typing import Any

import pytest


class MockModelHolder:
    """Mirror of mlx_whisper.transcribe.ModelHolder's class-level cache."""

    model: Any = None
    model_path: str | None = None
    fail_with: Exception | None = None

    @classmethod
    def get_model(cls, model_path: str, dtype: Any) -> Any:
        """Cache a model object for the path, or raise the configured error."""
        if cls.fail_with is not None:
            raise cls.fail_with
        cls.model, cls.model_path = object(), model_path
        return cls.model


class MockMlxWhisper:
    """Records transcribe calls and returns mlx-whisper shaped dictionaries."""

    def __init__(self) -> None:
        """Start with a two-segment English result."""
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.output: dict[str, Any] = {
            "text": " Hello there. General Kenobi.",
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": " Hello there.",
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.01,
                    "compression_ratio": 1.1,
                    "words": [
                        {"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                        {"word": " there.", "start": 0.5, "end": 1.2, "probability": 0.8},
                    ],
                },
                {"start": 1.5, "end": 2.5, "text": " General Kenobi.", "avg_logprob": -0.4, "no_speech_prob": None},
            ],
        }

    def transcribe(self, audio: str, **kwargs: Any) -> dict[str, Any]:
        """Record the call and return the configured output."""
        self.calls.append({"audio": audio, **kwargs})
        if self.error is not None:
            raise self.error
        return self.output


def install(monkeypatch: pytest.MonkeyPatch) -> MockMlxWhisper:
    """Install mock mlx, mlx.core, mlx_whisper and mlx_whisper.transcribe modules and return the recorder."""
    recorder = MockMlxWhisper()
    MockModelHolder.model, MockModelHolder.model_path, MockModelHolder.fail_with = None, None, None
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.float16 = "float16"  # type: ignore[attr-defined]
    mlx.core = core  # type: ignore[attr-defined]
    package = types.ModuleType("mlx_whisper")
    package.transcribe = recorder.transcribe  # type: ignore[attr-defined]
    transcribe_module = types.ModuleType("mlx_whisper.transcribe")
    transcribe_module.ModelHolder = MockModelHolder  # type: ignore[attr-defined]
    for name, module in {
        "mlx": mlx,
        "mlx.core": core,
        "mlx_whisper": package,
        "mlx_whisper.transcribe": transcribe_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "app.adapters.stt.mlx_whisper.backend", raising=False)
    return recorder
