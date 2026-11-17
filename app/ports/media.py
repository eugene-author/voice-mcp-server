"""Media probing and conversion ports."""

from pathlib import Path
from typing import Protocol

from app.domain.media import MediaInfo


class MediaProbe(Protocol):
    """Reads facts about media files."""

    async def probe(self, path: Path) -> MediaInfo:
        """Return duration, codec and size facts, or raise MediaError."""
        ...


class MediaConverter(Protocol):
    """Converts audio between formats."""

    async def to_wav(self, source: Path, destination: Path, *, sample_rate: int, channels: int = 1) -> Path:
        """Decode any supported input to PCM WAV at the given rate."""
        ...

    async def encode(self, source_wav: Path, destination: Path, *, output_format: str) -> Path:
        """Encode a WAV file to wav, mp3, flac or ogg."""
        ...

    async def normalize_loudness(self, source_wav: Path, destination: Path) -> Path:
        """Write a loudness-normalized, peak-limited copy of a WAV file."""
        ...
