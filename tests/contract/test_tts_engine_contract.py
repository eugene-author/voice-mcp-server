"""TtsEngine contract: one chunk in, mono float32 samples out, with tolerance-based checks."""

import platform
import sys
import uuid
from collections.abc import Iterator

import numpy as np
import pytest

from app.adapters.tts.factory import build_tts_engine
from app.domain.audio import rms
from app.domain.errors import InferenceError
from app.domain.ids import VoiceVersionId
from app.domain.models import InstalledModel, ModelKind, TtsVariant
from app.domain.tts import CloneReference, SynthesisRequest, TtsBackendName
from app.ports.tts import TtsEngine
from tests.mocks.tts import MockTtsEngine
from tests.real_models import CLONING_CHECKPOINT, MODELS_ROOT, REFERENCE_AUDIO, REFERENCE_TEXT, is_installed

IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def _checkpoint(backend: str) -> InstalledModel:
    """Return the Base checkpoint record of a runtime."""
    return InstalledModel(
        name=CLONING_CHECKPOINT,
        kind=ModelKind.TTS,
        backend=backend,
        local_path=MODELS_ROOT / "tts" / backend / CLONING_CHECKPOINT,
        variant=TtsVariant.BASE,
    )


@pytest.fixture(
    params=[
        "mock",
        pytest.param(
            "torch",
            marks=[
                pytest.mark.models,
                pytest.mark.skipif(not is_installed("tts", "torch", CLONING_CHECKPOINT), reason="needs the checkpoint"),
            ],
        ),
        pytest.param(
            "mlx",
            marks=[
                pytest.mark.models,
                pytest.mark.mlx,
                pytest.mark.skipif(not IS_APPLE_SILICON, reason="needs Apple Silicon"),
                pytest.mark.skipif(not is_installed("tts", "mlx", CLONING_CHECKPOINT), reason="needs the checkpoint"),
            ],
        ),
    ]
)
def engine(request: pytest.FixtureRequest) -> Iterator[TtsEngine]:
    """Return each TTS implementation with the Base checkpoint loaded."""
    if request.param == "mock":
        implementation: TtsEngine = MockTtsEngine()
    else:
        implementation = build_tts_engine(TtsBackendName(request.param), device="cpu", conditioning_cache_size=2)
    implementation.load(_checkpoint("torch" if request.param == "mock" else request.param))
    yield implementation
    implementation.unload()


def test_clone_synthesis_returns_plausible_audio(engine: TtsEngine) -> None:
    """Return non-silent mono float32 audio of plausible duration."""
    reference = CloneReference(
        voice_version_id=VoiceVersionId(uuid.uuid4()),
        audio_path=REFERENCE_AUDIO,
        text=REFERENCE_TEXT,
    )
    audio = engine.synthesize(SynthesisRequest(text="Good morning, everyone.", language="english", voice=reference))
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert 0.3 < audio.duration_seconds < 10
    assert rms(audio.samples) > 1e-3


def test_synthesize_without_load_raises(engine: TtsEngine) -> None:
    """Raise InferenceError when nothing is loaded."""
    engine.unload()
    reference = CloneReference(VoiceVersionId(uuid.uuid4()), REFERENCE_AUDIO, "text")
    with pytest.raises(InferenceError):
        engine.synthesize(SynthesisRequest(text="Hi.", language="english", voice=reference))
