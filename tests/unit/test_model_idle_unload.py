"""Model preload and idle unload through ModelHost with mock engines."""

import asyncio
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.core.executor import InferenceRunner
from app.domain.errors import CapabilityUnavailableError, ModelNotInstalledError
from app.domain.ids import VoiceVersionId
from app.domain.models import ModelKind
from app.domain.stt import SttRequest, VadSettings
from app.domain.tts import CloneReference, SynthesisRequest
from app.domain.workers import ModelRef
from app.services.model_host import ModelHost
from app.workers.idle_unload import IdleUnloader
from tests.harness import DEFAULT_INSTALLED
from tests.mocks.denoise import MockDenoiser
from tests.mocks.models import MockModelCatalog
from tests.mocks.stt import MockSttBackend
from tests.mocks.tts import MockTtsEngine
from tests.waiting import eventually


class _Ticker:
    """Monotonic seconds the test advances by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current reading."""
        return self.now


@pytest.fixture
def ticker() -> _Ticker:
    """Return a hand-driven monotonic clock."""
    return _Ticker()


@pytest.fixture
def model_host(ticker: _Ticker) -> Iterator[ModelHost]:
    """Return a non-exclusive model_host over mock engines and a denoiser."""
    executor = InferenceRunner()
    yield ModelHost(
        stt=MockSttBackend(),
        tts=MockTtsEngine(),
        denoiser=MockDenoiser(),
        vad=None,
        catalog=MockModelCatalog(installed=DEFAULT_INSTALLED),
        executor=executor,
        exclusive=False,
        monotonic=ticker,
    )
    executor.shutdown()


async def _transcribe(model_host: ModelHost, tmp_path: Path) -> None:
    """Use the small STT model once."""
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
    """Use the Base checkpoint once."""
    voice = CloneReference(VoiceVersionId(uuid.uuid4()), Path("/workspace/ref.wav"), "reference words")
    request = SynthesisRequest(text="Hello.", language="english", voice=voice)
    await model_host.synthesize(model_host.tts_checkpoint("Qwen3-TTS-12Hz-0.6B-Base"), request)


async def test_unloads_only_models_idle_longer_than_the_limit(
    model_host: ModelHost, ticker: _Ticker, tmp_path: Path
) -> None:
    """Release the STT model unused for the idle time and keep the TTS checkpoint used since."""
    await _transcribe(model_host, tmp_path)
    ticker.now = 200
    await _synthesize(model_host)
    ticker.now = 330
    assert await model_host.unload_idle(idle_seconds=300) == [ModelKind.STT]
    assert model_host.loaded_models == {"stt": None, "tts": "Qwen3-TTS-12Hz-0.6B-Base"}
    ticker.now = 600
    assert await model_host.unload_idle(idle_seconds=300) == [ModelKind.TTS]
    assert model_host.loaded_models == {"stt": None, "tts": None}


async def test_unloaded_model_loads_again_on_next_use(model_host: ModelHost, ticker: _Ticker, tmp_path: Path) -> None:
    """Reload a model that was unloaded for idleness the next time a job needs it."""
    await _transcribe(model_host, tmp_path)
    ticker.now = 1000
    await model_host.unload_idle(idle_seconds=300)
    await _transcribe(model_host, tmp_path)
    stt = model_host.stt
    assert isinstance(stt, MockSttBackend)
    assert stt.load_count == 2
    assert model_host.loaded_models["stt"] == "small"


async def test_preload_loads_each_kind_without_running_it(model_host: ModelHost) -> None:
    """Load STT, TTS and the denoiser up front; nothing is transcribed or synthesized."""
    for ref in ("stt:small", "tts:Qwen3-TTS-12Hz-0.6B-Base", "denoise:deepfilternet3"):
        await model_host.preload(ModelRef.parse(ref))
    stt, tts = model_host.stt, model_host.tts
    assert isinstance(stt, MockSttBackend)
    assert isinstance(tts, MockTtsEngine)
    assert model_host.loaded_models == {"stt": "small", "tts": "Qwen3-TTS-12Hz-0.6B-Base"}
    assert stt.requests == []
    assert await model_host.unload_idle(idle_seconds=0) == [ModelKind.STT, ModelKind.TTS, ModelKind.DENOISE]


async def test_preload_rejects_missing_models(ticker: _Ticker) -> None:
    """Raise model_not_installed for an absent model and capability_unavailable without a denoiser."""
    executor = InferenceRunner()
    model_host = ModelHost(
        stt=MockSttBackend(),
        tts=None,
        denoiser=None,
        vad=None,
        catalog=MockModelCatalog(installed=DEFAULT_INSTALLED),
        executor=executor,
        exclusive=False,
        monotonic=ticker,
    )
    try:
        with pytest.raises(ModelNotInstalledError):
            await model_host.preload(ModelRef(ModelKind.STT, "large-v3"))
        with pytest.raises(CapabilityUnavailableError):
            await model_host.preload(ModelRef(ModelKind.DENOISE, "deepfilternet3"))
    finally:
        executor.shutdown()


@pytest.mark.parametrize("raw", ["stt", "gpu:large", "stt:", ":small", "tts large"])
def test_malformed_model_references_are_rejected(raw: str) -> None:
    """Require kind:name with a known kind."""
    with pytest.raises(ValueError, match="kind:name"):
        ModelRef.parse(raw)


async def test_idle_unloader_loop_frees_models(tmp_path: Path) -> None:
    """Unload an idle model from the background loop while the process keeps running."""
    executor = InferenceRunner()
    model_host = ModelHost(
        stt=MockSttBackend(),
        tts=None,
        denoiser=None,
        vad=None,
        catalog=MockModelCatalog(installed=DEFAULT_INSTALLED),
        executor=executor,
        exclusive=False,
    )
    try:
        await _transcribe(model_host, tmp_path)
        loop = asyncio.create_task(IdleUnloader(model_host=model_host, idle_seconds=0.05).run())
        try:
            await eventually(lambda: model_host.loaded_models["stt"] is None)
        finally:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)
    finally:
        executor.shutdown()
