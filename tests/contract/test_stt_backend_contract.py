"""SttBackend contract: every implementation must behave the same for services."""

import importlib
import platform
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.adapters.stt.ctranslate2.backend import CTranslate2Backend
from app.domain.errors import InferenceError
from app.domain.models import InstalledModel, ModelKind
from app.domain.stt import SttBackendName, SttRequest, SttRuntime, VadSettings
from app.ports.stt import SttBackend
from tests.mocks.stt import MockSttBackend
from tests.real_models import MODELS_ROOT, REFERENCE_AUDIO, installed_stt_model

CT2_MODEL = installed_stt_model("ctranslate2")
MLX_TINY = MODELS_ROOT / "stt" / "mlx" / "tiny"
IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def _model(path: Path) -> InstalledModel:
    """Return an installed-model record for a directory."""
    return InstalledModel(name="small", kind=ModelKind.STT, backend="ctranslate2", local_path=path)


@pytest.fixture(
    params=[
        "mock",
        pytest.param(
            "ctranslate2", marks=[pytest.mark.models, pytest.mark.skipif(CT2_MODEL is None, reason="no model")]
        ),
    ]
)
def backend(request: pytest.FixtureRequest) -> Iterator[SttBackend]:
    """Return each STT implementation, loaded."""
    model_dir = MODELS_ROOT / "stt" / "ctranslate2" / (CT2_MODEL or "small")
    if request.param == "mock":
        implementation: SttBackend = MockSttBackend(text="hello world")
    elif request.param == "mlx":
        implementation = importlib.import_module("app.adapters.stt.mlx_whisper.backend").MlxWhisperBackend()
        model_dir = MLX_TINY
    else:
        implementation = CTranslate2Backend(SttRuntime(SttBackendName.CTRANSLATE2, "cpu", "int8"))
    implementation.load(_model(model_dir))
    yield implementation
    implementation.unload()


def _request(path: Path, **overrides: object) -> SttRequest:
    """Return a default request for a prepared WAV."""
    values: dict[str, object] = {
        "audio_path": path,
        "language": "en",
        "beam_size": 1,
        "temperature": 0.0,
        "word_timestamps": True,
        "vad": VadSettings(enabled=True),
    }
    values.update(overrides)
    return SttRequest(**values)  # type: ignore[arg-type]


def _speech_wav(tmp_path: Path) -> Path:
    """Return a 16 kHz mono WAV of the repository speech fixture."""
    return Path(shutil.copy(REFERENCE_AUDIO, tmp_path / "speech.wav"))


def test_transcribes_with_source_relative_ordered_timestamps(backend: SttBackend, tmp_path: Path) -> None:
    """Return non-empty text with ordered segments and words inside the audio duration."""
    result = backend.transcribe(_request(_speech_wav(tmp_path)))
    assert result.text.strip()
    assert result.language == "en"
    for segment in result.segments:
        assert 0 <= segment.start <= segment.end <= result.duration_seconds + 0.5
        for word in segment.words:
            assert segment.start - 0.5 <= word.start <= word.end <= segment.end + 0.5


def test_capabilities_and_runtime_are_declared(backend: SttBackend) -> None:
    """Declare capabilities and a runtime naming the backend."""
    assert backend.runtime.backend is backend.name
    assert isinstance(backend.capabilities.beam_search, bool)


def test_transcribe_without_load_raises(backend: SttBackend, tmp_path: Path) -> None:
    """Raise InferenceError when no model is loaded."""
    backend.unload()
    with pytest.raises(InferenceError):
        backend.transcribe(_request(_speech_wav(tmp_path)))
