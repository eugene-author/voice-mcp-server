"""FFmpeg media tool and local asset publication on the real filesystem."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.adapters.media.ffmpeg.tool import FfmpegMediaTool
from app.domain.errors import MediaError
from tests.conftest import write_tone

pytestmark = pytest.mark.integration


@pytest.fixture
def media() -> FfmpegMediaTool:
    """Return the FFmpeg tool from PATH."""
    return FfmpegMediaTool.from_path(probe_timeout_seconds=10, convert_timeout_seconds=60)


async def test_probe_decode_encode_and_normalize(media: FfmpegMediaTool, tmp_path: Path) -> None:
    """Probe, resample, encode every output format and normalize loudness."""
    source = write_tone(tmp_path / "in.wav", seconds=1.5, sample_rate=44100)
    info = await media.probe(source)
    assert info.duration_seconds == pytest.approx(1.5, abs=0.05)
    assert info.sample_rate == 44100
    wav = await media.to_wav(source, tmp_path / "16k.wav", sample_rate=16000)
    samples, rate = sf.read(str(wav))
    assert rate == 16000
    assert samples.ndim == 1
    for output_format in ("mp3", "flac", "ogg", "wav"):
        encoded = await media.encode(wav, tmp_path / f"out.{output_format}", output_format=output_format)
        assert (await media.probe(encoded)).duration_seconds == pytest.approx(1.5, abs=0.1)
    normalized = await media.normalize_loudness(wav, tmp_path / "norm.wav")
    level, normalized_rate = sf.read(str(normalized))
    assert normalized_rate == 16000
    assert np.max(np.abs(level)) <= 0.9


async def test_corrupt_input_raises_media_error(media: FfmpegMediaTool, tmp_path: Path) -> None:
    """Translate ffprobe failures to MediaError and remove partial outputs."""
    corrupt = tmp_path / "bad.mp3"
    corrupt.write_bytes(b"not audio at all")
    with pytest.raises(MediaError):
        await media.probe(corrupt)
    with pytest.raises(MediaError):
        await media.to_wav(corrupt, tmp_path / "out.wav", sample_rate=16000)
    assert not (tmp_path / "out.wav").exists()
