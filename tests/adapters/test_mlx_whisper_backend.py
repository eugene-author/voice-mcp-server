"""MlxWhisperBackend argument mapping, output normalization and error translation with mock MLX modules."""

import importlib
from pathlib import Path
from typing import Any

import pytest

from app.domain.errors import InferenceError
from app.domain.models import InstalledModel, ModelKind
from app.domain.stt import SttBackendName, SttRequest, VadSettings
from tests.mocks import mlx_whisper as mock_mlx

MODEL = InstalledModel(name="small", kind=ModelKind.STT, backend="mlx", local_path=Path("/models/stt/mlx/small"))


def _request(**overrides: object) -> SttRequest:
    """Return a request with mlx-compatible options."""
    values: dict[str, object] = {
        "audio_path": Path("/workspace/speech.wav"),
        "language": "en",
        "beam_size": 1,
        "temperature": 0.0,
        "word_timestamps": True,
        "vad": VadSettings(enabled=False),
        "initial_prompt": "Kenobi",
    }
    values.update(overrides)
    return SttRequest(**values)  # type: ignore[arg-type]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> mock_mlx.MockMlxWhisper:
    """Install mock MLX modules."""
    return mock_mlx.install(monkeypatch)


def _backend() -> Any:
    """Import the adapter against the mock modules and build it; its module needs the mocks, so it is untyped here."""
    module = importlib.import_module("app.adapters.stt.mlx_whisper.backend")
    return module.MlxWhisperBackend()


def test_declares_mlx_capabilities_and_runtime(recorder: mock_mlx.MockMlxWhisper) -> None:
    """Declare greedy decoding, no hotwords, no native VAD and a Metal runtime."""
    backend = _backend()
    capabilities = backend.capabilities
    assert backend.name is SttBackendName.MLX
    assert not capabilities.beam_search
    assert not capabilities.hotwords
    assert not capabilities.native_vad
    assert backend.runtime.device == "metal"


def test_maps_arguments_and_normalizes_output(recorder: mock_mlx.MockMlxWhisper) -> None:
    """Pass the local model directory and options, and map segments and words to domain types."""
    backend = _backend()
    backend.load(MODEL)
    result = backend.transcribe(_request())
    call = recorder.calls[0]
    assert call["path_or_hf_repo"] == str(MODEL.local_path)
    assert call["initial_prompt"] == "Kenobi"
    assert call["word_timestamps"] is True
    assert "beam_size" not in call
    assert result.text == "Hello there. General Kenobi."
    assert result.language_probability is None
    assert result.segments[0].words[0].text == "Hello"
    assert result.segments[1].no_speech_prob is None
    assert result.duration_seconds == pytest.approx(2.5)


def test_requires_load_and_translates_errors(recorder: mock_mlx.MockMlxWhisper) -> None:
    """Raise InferenceError before loading and for runtime failures."""
    backend = _backend()
    with pytest.raises(InferenceError):
        backend.transcribe(_request())
    backend.load(MODEL)
    recorder.error = RuntimeError("metal failure")
    with pytest.raises(InferenceError):
        backend.transcribe(_request())


def test_load_failure_and_unload(recorder: mock_mlx.MockMlxWhisper) -> None:
    """Translate load errors and clear the model holder on unload."""
    backend = _backend()
    mock_mlx.MockModelHolder.fail_with = OSError("missing weights")
    with pytest.raises(InferenceError):
        backend.load(MODEL)
    mock_mlx.MockModelHolder.fail_with = None
    backend.load(MODEL)
    backend.unload()
    assert mock_mlx.MockModelHolder.model is None
