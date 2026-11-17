"""Media probe and converter that write simple WAV files without FFmpeg."""

import asyncio
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from app.domain.errors import MediaError
from app.domain.media import MediaInfo


class MockMedia:
    """Implements MediaProbe and MediaConverter with soundfile."""

    def __init__(
        self, *, duration_seconds: float = 2.0, fail_probe: bool = False, failing_names: frozenset[str] = frozenset()
    ) -> None:
        """Configure the probed duration, a global failure mode and file names whose probe fails."""
        self.duration_seconds = duration_seconds
        self.fail_probe = fail_probe
        self.failing_names = failing_names
        self.encoded: list[tuple[Path, str]] = []

    async def probe(self, path: Path) -> MediaInfo:
        """Return the configured duration."""
        if self.fail_probe or path.name in self.failing_names:
            raise MediaError("could not read media information")
        return MediaInfo(
            duration_seconds=self.duration_seconds,
            format_name="wav",
            codec="pcm_s16le",
            sample_rate=16000,
            channels=1,
            size_bytes=(await asyncio.to_thread(path.stat)).st_size,
        )

    async def to_wav(self, source: Path, destination: Path, *, sample_rate: int, channels: int = 1) -> Path:
        """Write a tone of the configured duration at the requested rate."""
        length = int(self.duration_seconds * sample_rate)
        tone = 0.2 * np.sin(2 * np.pi * 220 * np.arange(length) / sample_rate)
        sf.write(str(destination), tone.astype(np.float32), sample_rate)
        return destination

    async def encode(self, source_wav: Path, destination: Path, *, output_format: str) -> Path:
        """Copy the WAV and record the format."""
        self.encoded.append((destination, output_format))
        shutil.copyfile(source_wav, destination)
        return destination

    async def normalize_loudness(self, source_wav: Path, destination: Path) -> Path:
        """Copy the WAV unchanged."""
        shutil.copyfile(source_wav, destination)
        return destination
