"""Model host: lazy loads and eviction of idle models when only one may stay resident."""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.executor import InferenceRunner
from app.domain.ids import VoiceVersionId
from app.domain.models import InstalledModel
from app.domain.stt import SttRequest, VadSettings
from app.domain.tts import CloneReference, SynthesisRequest
from app.services.model_host import ModelHost
from tests.harness import DEFAULT_INSTALLED
from tests.mocks.denoise import MockDenoiser
from tests.mocks.models import MockModelCatalog
from tests.mocks.stt import MockSttBackend
from tests.mocks.tts import MockTtsEngine

type HostedModels = tuple[ModelHost, MockSttBackend, MockTtsEngine, MockDenoiser]


def _build(*, exclusive: bool) -> HostedModels:
    """Return a model_host over mock runtimes with the default installed models."""
    stt, tts, denoiser = MockSttBackend(), MockTtsEngine(), MockDenoiser()
    model_host = ModelHost(
        stt=stt,
        tts=tts,
        denoiser=denoiser,
        vad=None,
        catalog=MockModelCatalog(installed=DEFAULT_INSTALLED),
        executor=InferenceRunner(),
        exclusive=exclusive,
    )
    return model_host, stt, tts, denoiser


@pytest.fixture
def exclusive() -> Iterator[HostedModels]:
    """Return an exclusive model_host, shut down after the test."""
    built = _build(exclusive=True)
    yield built
    built[0]._executor.shutdown()


def _record_residents_at_denoiser_load(stt: MockSttBackend, tts: MockTtsEngine, denoiser: MockDenoiser) -> list[str]:
    """Wrap the denoiser's load to record which models were still resident when it ran."""
    residents: list[str] = []
    original = denoiser.load

    def load(model: InstalledModel) -> None:
        """Record resident STT/TTS models, then load."""
        residents.extend(name for name, runtime in (("stt", stt), ("tts", tts)) if runtime.loaded is not None)
        original(model)

    denoiser.load = load  # type: ignore[method-assign]
    return residents


async def _denoise(model_host: ModelHost, tmp_path: Path) -> None:
    """Denoise a small source file."""
    source = tmp_path / "a.wav"
    source.write_bytes(b"RIFF")
    await model_host.denoise(source, tmp_path / "b.wav")


async def _transcribe(model_host: ModelHost, tmp_path: Path) -> None:
    """Load the small STT model by transcribing once."""
    request = SttRequest(
        audio_path=tmp_path / "in.wav",
        language="en",
        beam_size=1,
        temperature=0.0,
        word_timestamps=False,
        vad=VadSettings(),
    )
    await model_host.transcribe(model_host.stt_model("small"), request)


async def _synthesize(model_host: ModelHost) -> None:
    """Load the Base checkpoint by synthesizing once."""
    reference = CloneReference(VoiceVersionId(uuid.uuid4()), Path("/workspace/ref.wav"), "reference words")
    request = SynthesisRequest(text="Hello.", language="english", voice=reference)
    await model_host.synthesize(model_host.tts_checkpoint("Qwen3-TTS-12Hz-0.6B-Base"), request)


async def test_exclusive_denoise_unloads_idle_stt_before_loading(exclusive: HostedModels, tmp_path: Path) -> None:
    """Unload the resident STT model before DeepFilterNet loads, like the STT and TTS paths do."""
    model_host, stt, tts, denoiser = exclusive
    await _transcribe(model_host, tmp_path)
    residents = _record_residents_at_denoiser_load(stt, tts, denoiser)
    await _denoise(model_host, tmp_path)
    assert residents == []
    assert stt.loaded is None
    assert model_host.loaded_models == {"stt": None, "tts": None}


async def test_exclusive_denoise_unloads_idle_tts_before_loading(exclusive: HostedModels, tmp_path: Path) -> None:
    """Unload the resident TTS checkpoint before DeepFilterNet loads."""
    model_host, stt, tts, denoiser = exclusive
    await _synthesize(model_host)
    residents = _record_residents_at_denoiser_load(stt, tts, denoiser)
    await _denoise(model_host, tmp_path)
    assert residents == []
    assert tts.loaded is None


async def test_shared_residency_keeps_models_when_denoising(tmp_path: Path) -> None:
    """Keep STT resident alongside the denoiser when several models may stay loaded."""
    model_host, stt, tts, denoiser = _build(exclusive=False)
    await _transcribe(model_host, tmp_path)
    residents = _record_residents_at_denoiser_load(stt, tts, denoiser)
    await _denoise(model_host, tmp_path)
    assert residents == ["stt"]
    assert stt.loaded is not None
    model_host._executor.shutdown()
