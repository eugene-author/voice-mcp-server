"""Real-model pipeline scenarios shared by the Linux CPU, macOS MLX and GPU model suites."""

import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.adapters.media.ffmpeg.tool import FfmpegMediaTool
from app.core.settings import load_settings
from app.domain.audio import rms
from app.domain.jobs import JobStatus, RegisterVoiceCommand, SynthesizeCommand, TranscribeCommand
from app.domain.speech_text import chunk_text, transcript_similarity
from app.domain.stt import SttOptions
from app.domain.tts import TtsOptions
from app.domain.voices import PRESET_VOICE_PREFIX, VoiceKind
from app.main import build_inference
from app.services.job_service import Submission
from tests.harness import OWNER, Harness, build_harness

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "var" / "models"
SAMPLES_ROOT = ROOT / "var" / "ci-samples"
FIXTURES = ROOT / "tests" / "fixtures" / "demo"
REFERENCE_AUDIO = FIXTURES / "jessica.wav"
REFERENCE_TEXT = (FIXTURES / "jessica.txt").read_text(encoding="utf-8").strip()
SYNTHESIS_TEXT = (
    "Good morning to everyone listening. The weather is calm and clear today. "
    "After lunch we will walk down to the harbour and watch the boats."
)
MAX_CHUNK_CHARS = 60
MIN_SIMILARITY = 0.6
STT_MODELS = ("tiny", "small")
CLONING_CHECKPOINT = "Qwen3-TTS-12Hz-0.6B-Base"
"""Default tts.cloning_checkpoint, installed by the model CI jobs."""

PRESET_CHECKPOINT = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
"""Default tts.preset_checkpoint, installed by the model CI jobs."""

DENOISE_MODEL = "deepfilternet3"


def is_installed(kind: str, backend: str, name: str) -> bool:
    """Return whether models install has completed for this model."""
    return (MODELS_ROOT / kind / backend / name / "manifest.json").exists()


def installed_stt_model(backend: str) -> str | None:
    """Return the smallest installed STT model of a backend, if any."""
    return next((name for name in STT_MODELS if is_installed("stt", backend, name)), None)


def real_harness(tmp_path: Path, *, stt_model: str, overrides: dict[str, dict[str, Any]] | None = None) -> Harness:
    """Return the job pipeline wired to real models, real FFmpeg and in-memory persistence."""
    settings_overrides: dict[str, dict[str, Any]] = {
        "models": {"root": str(MODELS_ROOT)},
        "stt": {"model": stt_model},
    }
    for key, value in (overrides or {}).items():
        settings_overrides[key] = {**settings_overrides.get(key, {}), **value}
    inference = build_inference(
        load_settings(
            storage={"data_root": str(tmp_path / "data")}, server={"transport": "stdio"}, **settings_overrides
        )
    )
    return build_harness(
        tmp_path, inference=inference, media=FfmpegMediaTool.from_path(), settings_overrides=settings_overrides
    )


async def release(harness: Harness) -> None:
    """Unload every model and stop the inference thread."""
    await harness.model_host.unload_all()
    harness.executor.shutdown()


async def run_job(harness: Harness, command: Any) -> Submission:
    """Submit a command, process the queue and return the submission after asserting the job completed."""
    submission = await harness.jobs.submit(command, owner=OWNER)
    await harness.run_queued()
    record = harness.uow.db.jobs[submission.job_id]
    assert record.status is JobStatus.COMPLETED, record.error
    return submission


async def register_reference_voice(harness: Harness) -> str:
    """Register the fixture recording as an ICL voice and return its ID."""
    reference = Path(shutil.copy(REFERENCE_AUDIO, harness.inputs / REFERENCE_AUDIO.name))
    command = RegisterVoiceCommand(
        name="Fixture narrator",
        reference_audio_path=reference,
        consent_confirmed=True,
        reference_text=REFERENCE_TEXT,
        language="en",
    )
    submission = await run_job(harness, command)
    assert submission.voice_id is not None
    return str(submission.voice_id)


async def first_preset_voice(harness: Harness) -> str:
    """Return the voice ID of the first preset speaker of the installed CustomVoice checkpoint."""
    voices = await harness.voices.list_voices(owner=OWNER, language=None)
    presets = [voice.voice_id for voice in voices if voice.kind is VoiceKind.PRESET]
    assert presets, "the CustomVoice checkpoint lists no preset speakers"
    assert all(voice_id.startswith(PRESET_VOICE_PREFIX) for voice_id in presets)
    return presets[0]


async def synthesize_and_check(harness: Harness, voice_id: str, *, sample_name: str) -> Path:
    """Synthesize multi-chunk text, check every chunk and the stitched output, save it as a sample and return it."""
    command = SynthesizeCommand(
        text=SYNTHESIS_TEXT,
        voice_id=voice_id,
        language="en",
        options=TtsOptions(output_format="wav", max_chunk_chars=MAX_CHUNK_CHARS),
    )
    job_id = (await run_job(harness, command)).job_id
    metrics = harness.uow.db.metrics[job_id]
    expected_chunks = len(chunk_text(SYNTHESIS_TEXT, max_chars=MAX_CHUNK_CHARS))
    assert expected_chunks > 1
    assert metrics["chunks"] == expected_chunks
    assert metrics["failed_chunks"] == 0
    view = await harness.queries.get(job_id, owner=OWNER)
    runtime = harness.tts.name if harness.tts else "none"
    sample = SAMPLES_ROOT / f"{sys.platform}-{runtime}" / f"{sample_name}.wav"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(harness.store.blobs[view.files["wav"]])
    samples, rate = sf.read(str(sample), dtype="float32")
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    words = len(SYNTHESIS_TEXT.split())
    assert 0.15 * words < len(audio) / rate < 1.2 * words
    assert rms(audio) > 1e-3
    return sample


async def transcribe_text(harness: Harness, audio: Path, *, denoise: bool = False) -> str:
    """Transcribe a file through the job pipeline and return the full transcript text."""
    source = Path(shutil.copy(audio, harness.inputs / f"stt-{audio.name}"))
    command = TranscribeCommand(input_path=source, language="en", options=SttOptions(denoise=denoise))
    job_id = (await run_job(harness, command)).job_id
    if denoise:
        assert harness.uow.db.metrics[job_id].get("denoise") is True
    return harness.uow.db.transcripts[job_id].text


async def assert_round_trip(harness: Harness, sample: Path) -> None:
    """Check that transcribing the synthesized sample roughly recovers the input text."""
    text = await transcribe_text(harness, sample)
    assert transcript_similarity(SYNTHESIS_TEXT, text) > MIN_SIMILARITY, text


def noisy_fixture(directory: Path) -> Path:
    """Write the fixture speech mixed with white noise at about 10 dB SNR and return the path."""
    speech, rate = sf.read(str(REFERENCE_AUDIO), dtype="float32")
    noise = np.random.default_rng(0).standard_normal(len(speech)).astype(np.float32)
    mixed = speech + noise * (rms(speech) / rms(noise)) / np.sqrt(10)
    path = directory / "noisy_fox.wav"
    sf.write(str(path), mixed.astype(np.float32), rate)
    return path


async def assert_denoised_transcription(harness: Harness) -> None:
    """Transcribe noisy speech with denoising enabled and recover the fixture text."""
    text = await transcribe_text(harness, noisy_fixture(harness.inputs.parent), denoise=True)
    assert transcript_similarity(REFERENCE_TEXT, text) > MIN_SIMILARITY, text
