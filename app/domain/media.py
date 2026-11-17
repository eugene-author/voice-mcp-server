"""Media file facts reported by probing."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Probed facts about an audio file."""

    duration_seconds: float
    format_name: str
    codec: str | None
    sample_rate: int | None
    channels: int | None
    size_bytes: int
