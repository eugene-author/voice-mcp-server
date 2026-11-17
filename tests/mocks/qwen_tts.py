"""Stand-in for the qwen_tts module."""

import sys
import types
from typing import Any

import numpy as np
import pytest


class MockQwen3TTSModel:
    """Records calls like the official Qwen3TTSModel and returns a short tone."""

    instances: list[MockQwen3TTSModel] = []
    fail_load: Exception | None = None

    def __init__(self, path: str, kwargs: dict[str, Any]) -> None:
        """Store load arguments."""
        self.path = path
        self.load_kwargs = kwargs
        self.prompts_created = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> MockQwen3TTSModel:
        """Return a new recorder or raise the configured load error."""
        if cls.fail_load is not None:
            raise cls.fail_load
        instance = cls(path, kwargs)
        cls.instances.append(instance)
        return instance

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return a prompt token and count creations."""
        self.prompts_created += 1
        return [dict(kwargs)]

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[Any], int]:
        """Record and return one waveform at 24 kHz."""
        self.calls.append(("clone", kwargs))
        return [np.full(2400, 0.1, dtype=np.float32)], 24000

    def generate_custom_voice(self, **kwargs: Any) -> tuple[list[Any], int]:
        """Record and return one waveform at 24 kHz."""
        self.calls.append(("custom", kwargs))
        return [np.full(2400, 0.1, dtype=np.float32)], 24000


def install(monkeypatch: pytest.MonkeyPatch) -> type[MockQwen3TTSModel]:
    """Install a mock qwen_tts module and return the model class."""
    MockQwen3TTSModel.instances, MockQwen3TTSModel.fail_load = [], None
    module = types.ModuleType("qwen_tts")
    module.Qwen3TTSModel = MockQwen3TTSModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    monkeypatch.delitem(sys.modules, "app.adapters.tts.qwen_torch.engine", raising=False)
    return MockQwen3TTSModel
