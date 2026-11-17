"""Real faster-whisper, Qwen3-TTS (torch) and DeepFilterNet3 (onnxruntime) through the job pipeline on CPU."""

import platform
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.domain.stt import SttBackendName
from app.domain.tts import TtsBackendName
from tests.harness import Harness
from tests.real_models import (
    CLONING_CHECKPOINT,
    DENOISE_MODEL,
    PRESET_CHECKPOINT,
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

STT_MODEL = installed_stt_model("ctranslate2")
pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        sys.platform == "darwin" and platform.machine() == "arm64", reason="tests/mlx covers Apple Silicon"
    ),
    pytest.mark.skipif(STT_MODEL is None, reason="needs the tiny or small ctranslate2 model"),
]
needs_base = pytest.mark.skipif(not is_installed("tts", "torch", CLONING_CHECKPOINT), reason="needs the Base model")
needs_preset = pytest.mark.skipif(
    not is_installed("tts", "torch", PRESET_CHECKPOINT), reason="needs the CustomVoice model"
)
needs_denoise = pytest.mark.skipif(
    not is_installed("denoise", "deepfilternet", DENOISE_MODEL), reason="needs deepfilternet3"
)


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    """Return the CPU pipeline with real models, released after the test."""
    assert STT_MODEL is not None
    harness = real_harness(tmp_path, stt_model=STT_MODEL, overrides={"stt": {"device": "cpu"}})
    assert harness.stt.name is SttBackendName.CTRANSLATE2
    yield harness
    await release(harness)


@needs_base
async def test_cloned_voice_synthesis_round_trips(harness: Harness) -> None:
    """Register the fixture voice, synthesize several chunks with it and transcribe the result back."""
    assert harness.tts is not None
    assert harness.tts.name is TtsBackendName.TORCH
    voice_id = await register_reference_voice(harness)
    await assert_round_trip(harness, await synthesize_and_check(harness, voice_id, sample_name="cloned"))


@needs_preset
async def test_preset_speaker_synthesis_round_trips(harness: Harness) -> None:
    """Synthesize several chunks with a CustomVoice preset speaker and transcribe the result back."""
    voice_id = await first_preset_voice(harness)
    await assert_round_trip(harness, await synthesize_and_check(harness, voice_id, sample_name="preset"))


@needs_denoise
async def test_denoised_transcription_recovers_noisy_speech(harness: Harness) -> None:
    """Denoise noisy speech with DeepFilterNet3 on CPU before transcription."""
    assert harness.denoiser is not None
    assert harness.denoiser.device == "cpu"
    await assert_denoised_transcription(harness)
