"""Text-to-speech value types shared by every TTS runtime."""

from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from app.domain.ids import VoiceVersionId

TTS_LANGUAGES = {
    "auto": "auto",
    "en": "english",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "es": "spanish",
    "pt": "portuguese",
    "ru": "russian",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
}
"""ISO language codes accepted by tools, mapped to Qwen3-TTS language names."""

OUTPUT_FORMATS = ("wav", "mp3", "flac", "ogg")


class TtsBackendName(StrEnum):
    """Concrete TTS runtime in use."""

    TORCH = "torch"
    MLX = "mlx"


class TtsBackendChoice(StrEnum):
    """Configured TTS runtime preference."""

    AUTO = "auto"
    TORCH = "torch"
    MLX = "mlx"


class ConditioningMode(StrEnum):
    """How a cloned voice conditions the Base checkpoint."""

    ICL = "icl"
    XVECTOR = "xvector"


@dataclass(frozen=True, slots=True)
class TtsCapabilities:
    """Features a TTS runtime supports."""

    clone_icl: bool
    clone_xvector_only: bool
    preset_voices: bool
    reusable_clone_prompt: bool


@dataclass(frozen=True, slots=True)
class CloneReference:
    """Pinned voice version used to condition a cloned voice."""

    voice_version_id: VoiceVersionId
    audio_path: Path
    """Working copy of the pinned version's reference audio."""

    text: str | None


@dataclass(frozen=True, slots=True)
class PresetSpeaker:
    """Built-in CustomVoice speaker."""

    name: str


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """Synthesis of one text chunk in one voice."""

    text: str
    """One internal chunk; SynthesisService does the chunking."""

    language: str
    """Qwen3-TTS language name such as english or auto."""

    voice: CloneReference | PresetSpeaker


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    """Mono audio samples produced by a TTS runtime."""

    samples: npt.NDArray[np.float32]
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        """Return the audio duration."""
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0


@dataclass(frozen=True, slots=True)
class TtsOptions:
    """Optional synthesis options; None means not specified at this level."""

    output_format: str | None = None
    loudness_normalization: bool | None = None
    max_chunk_chars: int | None = None
    gap_ms: int | None = None

    def specified(self) -> frozenset[str]:
        """Return the names of options that are set."""
        return frozenset(item.name for item in fields(self) if getattr(self, item.name) is not None)


@dataclass(frozen=True, slots=True)
class ResolvedTtsOptions:
    """Effective synthesis options."""

    output_format: str = "wav"
    loudness_normalization: bool = True
    max_chunk_chars: int = 300
    gap_ms: int = 120

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild options from a snapshot."""
        return cls(**data)
