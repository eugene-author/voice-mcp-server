"""Pydantic contracts for optional tool option objects."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.stt import SttOptions
from app.domain.tts import TtsOptions


class SttOptionsInput(BaseModel):
    """Advanced transcription options; omitted values come from the preset or server defaults."""

    model_config = ConfigDict(extra="forbid")

    beam_size: int | None = Field(default=None, ge=1, le=10)
    temperature: float | None = Field(default=None, ge=0, le=1)
    word_timestamps: bool | None = None
    initial_prompt: str | None = Field(default=None, max_length=500, description="Short glossary or context.")
    hotwords: list[str] | None = Field(default=None, max_length=50)
    vad: bool | None = Field(default=None, description="Silero voice activity detection.")
    denoise: bool | None = Field(default=None, description="DeepFilterNet denoising before transcription.")
    condition_on_previous_text: bool | None = None
    no_speech_threshold: float | None = Field(default=None, ge=0, le=1)
    remove_fillers: bool | None = None
    remove_repetitions: bool | None = None

    def to_domain(self) -> SttOptions:
        """Return the domain options."""
        values = self.model_dump()
        values["hotwords"] = tuple(self.hotwords) if self.hotwords is not None else None
        return SttOptions(**values)


class TtsOptionsInput(BaseModel):
    """Advanced synthesis options."""

    model_config = ConfigDict(extra="forbid")

    output_format: Literal["wav", "mp3", "flac", "ogg"] | None = None
    loudness_normalization: bool | None = None
    max_chunk_chars: int | None = Field(default=None, ge=50, le=2000)
    gap_ms: int | None = Field(default=None, ge=0, le=2000)

    def to_domain(self) -> TtsOptions:
        """Return the domain options."""
        return TtsOptions(**self.model_dump())
