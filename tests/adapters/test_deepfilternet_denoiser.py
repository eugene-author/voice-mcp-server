"""DeepFilterNetDenoiser with a mock onnxruntime: local model file, framing, state and latency compensation."""

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from app.domain.errors import InferenceError, MediaError
from app.domain.models import InstalledModel, ModelKind
from tests.mocks import onnxruntime as mock_ort

MODEL = InstalledModel(
    name="deepfilternet3", kind=ModelKind.DENOISE, backend="deepfilternet", local_path=Path("/models/denoise/df3")
)


def _denoiser(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    """Import the adapter against the mock onnxruntime and build it; untyped because its module needs the mock."""
    providers = kwargs.pop("providers", ("CPUExecutionProvider",))
    mock_ort.install(monkeypatch, providers=providers)
    module = importlib.import_module("app.adapters.denoise.deepfilternet.denoiser")
    return module.DeepFilterNetDenoiser(**kwargs)


def test_loads_local_file_and_output_is_aligned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Load the installed ONNX file and return input-aligned audio of the same length."""
    denoiser = _denoiser(monkeypatch, device="cpu", attenuation_limit_db=12.0)
    with pytest.raises(InferenceError):
        denoiser.enhance(np.zeros(10, dtype=np.float32))
    denoiser.load(MODEL)
    signal = np.random.default_rng(1).standard_normal(48000 + 123).astype(np.float32)
    source = tmp_path / "in.wav"
    sf.write(str(source), signal, 48000, subtype="FLOAT")
    denoiser.denoise(source, tmp_path / "out.wav")
    output, rate = sf.read(str(tmp_path / "out.wav"), dtype="float32")
    session = mock_ort.MockInferenceSession.instances[0]
    assert session.path == str(MODEL.local_path / "denoiser_model.onnx")
    assert rate == 48000
    assert len(output) == len(signal)
    assert np.allclose(output, signal)
    assert set(session.attenuation) == {12.0}
    assert denoiser.device == "cpu"


def test_state_is_reset_per_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each file from a fresh state so audio does not leak between files."""
    denoiser = _denoiser(monkeypatch, device="cpu")
    denoiser.load(MODEL)
    first = denoiser.enhance(np.ones(4800, dtype=np.float32))
    second = denoiser.enhance(np.zeros(4800, dtype=np.float32))
    assert np.allclose(first, 1.0)
    assert np.allclose(second, 0.0)


def test_rejects_wrong_rate_and_stereo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Require 48 kHz mono input."""
    denoiser = _denoiser(monkeypatch, device="cpu")
    denoiser.load(MODEL)
    for name, data, rate in (("a.wav", np.zeros(1600), 16000), ("b.wav", np.zeros((4800, 2)), 48000)):
        sf.write(str(tmp_path / name), data, rate)
        with pytest.raises(MediaError):
            denoiser.denoise(tmp_path / name, tmp_path / "out.wav")


def test_cuda_only_when_provider_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report and use CUDA only when this onnxruntime build has the CUDA provider."""
    assert _denoiser(monkeypatch, device="cuda").device == "cpu"
    with_gpu = _denoiser(monkeypatch, device="cuda", providers=("CUDAExecutionProvider", "CPUExecutionProvider"))
    assert with_gpu.device == "cuda"
    with_gpu.load(MODEL)
    assert mock_ort.MockInferenceSession.instances[-1].providers[0] == "CUDAExecutionProvider"
