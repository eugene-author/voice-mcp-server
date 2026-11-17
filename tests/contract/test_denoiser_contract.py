"""Denoiser contract: same length, aligned output, noise removed, speech kept."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.adapters.denoise.factory import build_denoiser
from app.domain.models import InstalledModel, ModelKind
from app.ports.denoise import Denoiser
from tests.mocks.denoise import MockDenoiser

ROOT = Path(__file__).resolve().parents[2]
DF3 = ROOT / "var" / "models" / "denoise" / "deepfilternet" / "deepfilternet3"
FIXTURE = ROOT / "tests" / "fixtures" / "demo" / "jessica.wav"
MODEL = InstalledModel(name="deepfilternet3", kind=ModelKind.DENOISE, backend="deepfilternet", local_path=DF3)
RATE = 48000


@pytest.fixture(
    params=[
        "mock",
        pytest.param(
            "onnx",
            marks=[
                pytest.mark.models,
                pytest.mark.skipif(not (DF3 / "denoiser_model.onnx").exists(), reason="no model"),
            ],
        ),
    ]
)
def denoiser(request: pytest.FixtureRequest) -> Iterator[Denoiser]:
    """Return each denoiser, loaded."""
    implementation = MockDenoiser() if request.param == "mock" else build_denoiser(device="cpu", attenuation_limit_db=0)
    implementation.load(MODEL)
    yield implementation
    implementation.unload()


def _db(samples: np.ndarray) -> float:
    """Return the RMS level in dBFS."""
    return float(20 * np.log10(np.sqrt(np.mean(np.square(samples))) + 1e-12))


def _noisy_speech(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """Write speech followed by one second of silence, both with white noise, at 48 kHz."""
    speech, rate = sf.read(str(FIXTURE), dtype="float32")
    upsampled = np.interp(np.arange(len(speech) * RATE // rate) * rate / RATE, np.arange(len(speech)), speech)
    clean = np.concatenate([upsampled, np.zeros(RATE)]).astype(np.float32)
    noisy = (clean + 0.03 * np.random.default_rng(0).standard_normal(len(clean))).astype(np.float32)
    path = tmp_path / "noisy.wav"
    sf.write(str(path), noisy, RATE, subtype="FLOAT")
    return path, clean, noisy


def test_preserves_length_and_level(denoiser: Denoiser, tmp_path: Path) -> None:
    """Keep the length and the speech level within 3 dB."""
    source, clean, _ = _noisy_speech(tmp_path)
    denoiser.denoise(source, tmp_path / "out.wav")
    output, rate = sf.read(str(tmp_path / "out.wav"), dtype="float32")
    speech = slice(0, len(clean) - RATE)
    assert rate == RATE
    assert len(output) == len(clean)
    assert abs(_db(output[speech]) - _db(clean[speech])) < 3


@pytest.mark.models
@pytest.mark.skipif(not (DF3 / "denoiser_model.onnx").exists(), reason="no model")
def test_real_model_removes_noise_without_shifting_speech(tmp_path: Path) -> None:
    """Attenuate a noise-only region by at least 20 dB and keep speech aligned with the input."""
    denoiser = build_denoiser(device="cpu", attenuation_limit_db=0)
    denoiser.load(MODEL)
    source, clean, noisy = _noisy_speech(tmp_path)
    denoiser.denoise(source, tmp_path / "out.wav")
    output, _ = sf.read(str(tmp_path / "out.wav"), dtype="float32")
    tail = slice(len(clean) - RATE // 2, len(clean))
    assert _db(output[tail]) < _db(noisy[tail]) - 20
    window = slice(RATE, 3 * RATE)
    correlation = np.correlate(output[window], clean[window], "full")
    assert abs(int(np.argmax(correlation)) - (2 * RATE - 1)) <= 48
