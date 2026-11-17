"""Real mlx-whisper transcription with standalone Silero VAD on macOS arm64 runners."""

import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.domain.speech_text import transcript_similarity
from app.domain.stt import SttBackendName, SttRequest, VadSettings
from app.domain.timeline import SpeechTimeline, map_to_source
from app.main import build_inference
from tests.mlx.conftest import FIXTURES, mlx_settings

pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(not (sys.platform == "darwin" and platform.machine() == "arm64"), reason="needs Apple Silicon"),
]


async def test_tiny_transcribes_fixture_with_vad_and_source_timestamps(tmp_path: Path) -> None:
    """Select mlx, run VAD then mlx-whisper, and return source-relative timestamps matching the ground truth."""
    inference = build_inference(mlx_settings(tmp_path))
    assert inference.stt.name is SttBackendName.MLX
    samples, rate = sf.read(str(FIXTURES / "jessica.wav"), dtype="float32")
    audio = np.asarray(samples, dtype=np.float32)
    regions = await inference.model_host.detect_speech(audio, rate, VadSettings())
    timeline = SpeechTimeline(regions, merge_gap_seconds=0.3)
    speech = tmp_path / "speech.wav"
    sf.write(str(speech), timeline.extract(audio, rate), rate)
    request = SttRequest(
        audio_path=speech,
        language="en",
        beam_size=1,
        temperature=0.0,
        word_timestamps=True,
        vad=VadSettings(enabled=False),
    )
    result = map_to_source(
        await inference.model_host.transcribe(inference.model_host.stt_model("tiny"), request), timeline
    )
    assert transcript_similarity((FIXTURES / "jessica.txt").read_text(), result.text) > 0.6
    assert result.segments[0].start >= timeline.regions[0].start - 0.1
    inference.executor.shutdown()
