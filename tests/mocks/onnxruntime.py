"""Stand-in for onnxruntime that emulates DeepFilterNet3's streaming interface and latency."""

import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

FRAME = 480
LATENCY = 1440
"""STFT delay (480) plus two lookahead frames (960), as in the real graph."""

STATE_SIZE = LATENCY + FRAME


@dataclass
class MockNodeArg:
    """Graph input description."""

    name: str
    shape: list[int]


class MockSessionOptions:
    """Accepts thread settings."""

    intra_op_num_threads = 0
    inter_op_num_threads = 0


class MockInferenceSession:
    """Echoes input delayed by LATENCY samples, keeping the delay line in the state vector."""

    instances: list[MockInferenceSession] = []

    def __init__(self, path: str, sess_options: Any = None, providers: list[str] | None = None) -> None:
        """Record construction arguments."""
        self.path = path
        self.providers = providers or []
        self.frames = 0
        self.attenuation: list[float] = []
        MockInferenceSession.instances.append(self)

    def get_inputs(self) -> list[MockNodeArg]:
        """Return the real model's input layout with this mock's state size."""
        return [
            MockNodeArg("input_frame", [FRAME]),
            MockNodeArg("states", [STATE_SIZE]),
            MockNodeArg("atten_lim_db", []),
        ]

    def get_providers(self) -> list[str]:
        """Return the providers in use."""
        return self.providers

    def run(self, outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Shift the frame into the delay line and emit the samples from LATENCY ago."""
        assert feed["input_frame"].shape == (FRAME,)
        assert feed["states"].shape == (STATE_SIZE,)
        self.frames += 1
        self.attenuation.append(float(feed["atten_lim_db"]))
        new_states = np.concatenate([feed["states"][FRAME:], feed["input_frame"]]).astype(np.float32)
        return [new_states[:FRAME].copy(), new_states, np.zeros(1, dtype=np.float32)]


def install(
    monkeypatch: pytest.MonkeyPatch, *, providers: tuple[str, ...] = ("CPUExecutionProvider",)
) -> type[MockInferenceSession]:
    """Install a mock onnxruntime module and return the session class."""
    MockInferenceSession.instances = []
    module = types.ModuleType("onnxruntime")
    module.InferenceSession = MockInferenceSession  # type: ignore[attr-defined]
    module.SessionOptions = MockSessionOptions  # type: ignore[attr-defined]
    module.get_available_providers = lambda: list(providers)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    monkeypatch.delitem(sys.modules, "app.adapters.denoise.deepfilternet.denoiser", raising=False)
    return MockInferenceSession
