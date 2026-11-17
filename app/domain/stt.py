"""Speech-to-text value types shared by every STT backend."""

from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

STT_MODEL_NAMES = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")


class SttBackendName(StrEnum):
    """Concrete STT backend in use."""

    CTRANSLATE2 = "ctranslate2"
    MLX = "mlx"


class SttBackendChoice(StrEnum):
    """Configured STT backend preference."""

    AUTO = "auto"
    CTRANSLATE2 = "ctranslate2"
    MLX = "mlx"


@dataclass(frozen=True, slots=True)
class SttCapabilities:
    """Options an STT backend can honor."""

    beam_search: bool
    hotwords: bool
    initial_prompt: bool
    native_vad: bool
    word_timestamps: bool
    language_probability: bool


@dataclass(frozen=True, slots=True)
class SttRuntime:
    """Device and precision an STT backend runs with."""

    backend: SttBackendName
    device: Literal["cuda", "cpu", "metal"]
    compute_type: str


@dataclass(frozen=True, slots=True)
class VadSettings:
    """Silero VAD parameters."""

    enabled: bool = True
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 250
    merge_gap_ms: int = 300

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild settings from a snapshot."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SttOptions:
    """Optional transcription options; None means not specified at this level."""

    beam_size: int | None = None
    temperature: float | None = None
    word_timestamps: bool | None = None
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] | None = None
    vad: bool | None = None
    denoise: bool | None = None
    condition_on_previous_text: bool | None = None
    no_speech_threshold: float | None = None
    remove_fillers: bool | None = None
    remove_repetitions: bool | None = None

    def merged_over(self, base: SttOptions) -> SttOptions:
        """Return these options with unset values taken from base."""
        overrides = {
            item.name: getattr(self, item.name) for item in fields(self) if getattr(self, item.name) is not None
        }
        return replace(base, **overrides)

    def specified(self) -> frozenset[str]:
        """Return the names of options that are set."""
        return frozenset(item.name for item in fields(self) if getattr(self, item.name) is not None)


@dataclass(frozen=True, slots=True)
class Substitution:
    """Preset or default value replaced because the backend cannot honor it."""

    option: str
    requested: Any
    used: Any
    reason: str

    def describe(self) -> str:
        """Return a human-readable warning line."""
        return f"{self.option} {self.requested!r} {self.reason}; used {self.used!r}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot."""
        return {"option": self.option, "requested": self.requested, "used": self.used, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ResolvedSttOptions:
    """Effective transcription options after defaults, presets and capability checks."""

    beam_size: int = 5
    temperature: float = 0.0
    word_timestamps: bool = True
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    vad: VadSettings = field(default_factory=VadSettings)
    denoise: bool = False
    condition_on_previous_text: bool = True
    no_speech_threshold: float = 0.6
    remove_fillers: bool = False
    remove_repetitions: bool = False
    segment_flags: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot."""
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        data["hotwords"] = list(self.hotwords)
        data["vad"] = self.vad.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild options from a snapshot."""
        values = dict(data)
        values["hotwords"] = tuple(values.get("hotwords", ()))
        values["vad"] = VadSettings.from_dict(values.get("vad", {}))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SttRequest:
    """Resolved transcription request for one prepared audio file."""

    audio_path: Path
    """Normalized 16 kHz mono WAV in the job workspace."""

    language: str | None
    beam_size: int
    temperature: float
    word_timestamps: bool
    vad: VadSettings
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    condition_on_previous_text: bool = True
    no_speech_threshold: float = 0.6


@dataclass(frozen=True, slots=True)
class Word:
    """Recognized word with source-relative timing."""

    text: str
    start: float
    end: float
    probability: float | None


@dataclass(frozen=True, slots=True)
class Segment:
    """Recognized segment with optional quality signals."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Backend-independent transcription output with source-relative timestamps."""

    segments: tuple[Segment, ...]
    language: str
    language_probability: float | None
    duration_seconds: float

    @property
    def text(self) -> str:
        """Return the full transcript text."""
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())
