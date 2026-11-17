"""Real Qwen3-TTS on mlx-audio, DeepFilterNet3 on CPU and MLX registration rules through the job pipeline."""

import platform
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.domain.errors import InvalidRequestError
from app.domain.jobs import RegisterVoiceCommand
from app.domain.stt import SttBackendName
from app.domain.tts import TtsBackendName
from tests.harness import OWNER, Harness
from tests.real_models import (
    CLONING_CHECKPOINT,
    DENOISE_MODEL,
    PRESET_CHECKPOINT,
    REFERENCE_AUDIO,
    assert_denoised_transcription,
    assert_round_trip,
    first_preset_voice,
    installed_stt_model,
    is_installed,
    real_harness,
    register_reference_voice,
    release,
    synthesize_and_check,
)

STT_MODEL = installed_stt_model("mlx")
pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(not (sys.platform == "darwin" and platform.machine() == "arm64"), reason="needs Apple Silicon"),
    pytest.mark.skipif(STT_MODEL is None, reason="needs the tiny or small mlx model"),
]


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    """Return the MLX pipeline with real models, released after the test."""
    assert STT_MODEL is not None
    harness = real_harness(tmp_path, stt_model=STT_MODEL)
    assert harness.stt.name is SttBackendName.MLX
    assert harness.tts is not None
    assert harness.tts.name is TtsBackendName.MLX
    yield harness
    await release(harness)


@pytest.mark.skipif(not is_installed("tts", "mlx", CLONING_CHECKPOINT), reason="needs the Base model")
async def test_base_cloning_round_trips(harness: Harness) -> None:
    """Register the fixture voice with its transcript, synthesize several chunks and transcribe them back."""
    voice_id = await register_reference_voice(harness)
    await assert_round_trip(harness, await synthesize_and_check(harness, voice_id, sample_name="cloned"))


@pytest.mark.skipif(not is_installed("tts", "mlx", PRESET_CHECKPOINT), reason="needs the CustomVoice model")
async def test_custom_voice_preset_round_trips(harness: Harness) -> None:
    """Synthesize several chunks with a preset speaker and transcribe them back."""
    voice_id = await first_preset_voice(harness)
    await assert_round_trip(harness, await synthesize_and_check(harness, voice_id, sample_name="preset"))


@pytest.mark.skipif(not is_installed("denoise", "deepfilternet", DENOISE_MODEL), reason="needs deepfilternet3")
async def test_cpu_denoise_pass(harness: Harness) -> None:
    """Run DeepFilterNet3 on CPU before mlx-whisper and recover the noisy fixture text."""
    assert harness.denoiser is not None
    assert harness.denoiser.device == "cpu"
    await assert_denoised_transcription(harness)


async def test_xvector_only_registration_is_invalid_request(harness: Harness) -> None:
    """Reject a registration without reference_text because mlx-audio clones with ICL only."""
    reference = Path(shutil.copy(REFERENCE_AUDIO, harness.inputs / REFERENCE_AUDIO.name))
    command = RegisterVoiceCommand(name="No transcript", reference_audio_path=reference, consent_confirmed=True)
    with pytest.raises(InvalidRequestError, match="reference_text is required") as raised:
        await harness.jobs.submit(command, owner=OWNER)
    assert raised.value.code == "invalid_request"
    assert not harness.uow.db.jobs
