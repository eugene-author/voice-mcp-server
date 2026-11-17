"""CTranslate2, Qwen3-TTS and DeepFilterNet3 on CUDA through the job pipeline, with real models."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.adapters.host.probe import SystemHostProbe
from app.domain.speech_text import transcript_similarity
from tests.harness import Harness
from tests.real_models import (
    CLONING_CHECKPOINT,
    DENOISE_MODEL,
    MIN_SIMILARITY,
    REFERENCE_AUDIO,
    REFERENCE_TEXT,
    assert_denoised_transcription,
    assert_round_trip,
    installed_stt_model,
    is_installed,
    real_harness,
    register_reference_voice,
    release,
    synthesize_and_check,
    transcribe_text,
)

STT_MODEL = installed_stt_model("ctranslate2")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(STT_MODEL is None, reason="needs the tiny or small ctranslate2 model"),
]


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    """Return the pipeline forced onto CUDA, skipping when the host has no usable CUDA device."""
    host = SystemHostProbe().probe()
    if not (host.cuda_device_count and host.torch_cuda_available):
        pytest.skip("needs a CUDA device visible to CTranslate2 and PyTorch")
    assert STT_MODEL is not None
    overrides = {"stt": {"device": "cuda"}, "tts": {"allow_cpu": False}}
    harness = real_harness(tmp_path, stt_model=STT_MODEL, overrides=overrides)
    yield harness
    await release(harness)


async def test_ctranslate2_transcribes_on_cuda(harness: Harness) -> None:
    """Transcribe the fixture with faster-whisper on the GPU."""
    assert harness.stt.runtime.device == "cuda"
    text = await transcribe_text(harness, REFERENCE_AUDIO)
    assert transcript_similarity(REFERENCE_TEXT, text) > MIN_SIMILARITY, text


@pytest.mark.skipif(not is_installed("tts", "torch", CLONING_CHECKPOINT), reason="needs the Base model")
async def test_qwen3_tts_clones_and_synthesizes_on_cuda(harness: Harness) -> None:
    """Register a voice and synthesize several chunks with Qwen3-TTS on the GPU."""
    assert harness.tts is not None
    assert harness.tts.device.startswith("cuda")
    voice_id = await register_reference_voice(harness)
    await assert_round_trip(harness, await synthesize_and_check(harness, voice_id, sample_name="cloned-cuda"))


@pytest.mark.skipif(not is_installed("denoise", "deepfilternet", DENOISE_MODEL), reason="needs deepfilternet3")
async def test_denoise_uses_cuda_execution_provider(harness: Harness) -> None:
    """Denoise on onnxruntime's CUDAExecutionProvider when onnxruntime-gpu provides it."""
    assert harness.denoiser is not None
    if harness.denoiser.device != "cuda":
        pytest.skip("onnxruntime has no CUDAExecutionProvider on this host")
    await assert_denoised_transcription(harness)
