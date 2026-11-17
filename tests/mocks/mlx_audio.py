"""Stand-in for the mlx_audio package."""

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@dataclass
class MockGenerationResult:
    """One generated segment, shaped like mlx-audio results."""

    audio: Any


class MockMlxAudioModel:
    """Records generate calls with the keyword signatures of mlx-audio 0.6.4 Qwen3-TTS and yields segments."""

    sample_rate = 24000

    def __init__(self, path: Path) -> None:
        """Store the model path."""
        self.path = path
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate(
        self,
        text: str,
        voice: str | None = None,
        instruct: str | None = None,
        lang_code: str = "auto",
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> list[MockGenerationResult]:
        """Record an ICL clone call."""
        self.calls.append(
            ("generate", {"text": text, "lang_code": lang_code, "ref_audio": ref_audio, "ref_text": ref_text})
        )
        return [MockGenerationResult([0.1] * 100), MockGenerationResult(np.full(50, 0.2, dtype=np.float32))]

    def generate_custom_voice(
        self, text: str, speaker: str, language: str = "auto", instruct: str | None = None
    ) -> list[MockGenerationResult]:
        """Record a preset speaker call."""
        self.calls.append(("custom", {"text": text, "speaker": speaker, "language": language}))
        return [MockGenerationResult([0.1] * 100)]


class MockMlxAudio:
    """Holds the models loaded through the mock load_model."""

    def __init__(self) -> None:
        """Start without models."""
        self.models: list[MockMlxAudioModel] = []

    def load_model(self, model_path: Path, lazy: bool = False, strict: bool = True) -> MockMlxAudioModel:
        """Return a new mock model."""
        model = MockMlxAudioModel(model_path)
        self.models.append(model)
        return model


def install(monkeypatch: pytest.MonkeyPatch) -> MockMlxAudio:
    """Install mock mlx_audio, mlx_audio.tts and mlx_audio.tts.utils modules and return the recorder."""
    recorder = MockMlxAudio()
    utils = types.ModuleType("mlx_audio.tts.utils")
    utils.load_model = recorder.load_model  # type: ignore[attr-defined]
    for name, module in {
        "mlx_audio": types.ModuleType("mlx_audio"),
        "mlx_audio.tts": types.ModuleType("mlx_audio.tts"),
        "mlx_audio.tts.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "app.adapters.tts.qwen_mlx.engine", raising=False)
    return recorder
