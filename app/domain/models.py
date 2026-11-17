"""Model registry entries and installed models."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ModelKind(StrEnum):
    """Category of model weights."""

    STT = "stt"
    TTS = "tts"
    DENOISE = "denoise"


class TtsVariant(StrEnum):
    """Qwen3-TTS checkpoint variant, which decides what a checkpoint can do."""

    BASE = "base"
    CUSTOM_VOICE = "custom_voice"


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """Registry entry for one logical model on one backend or runtime."""

    name: str
    kind: ModelKind
    backend: str
    """STT backend, TTS runtime or denoise runtime the weights are built for."""

    source: str
    """Hugging Face repository ID or release identifier."""

    approx_size_mb: int
    min_memory_mb: int
    """Rough memory guidance for operators choosing a model; descriptive only, not enforced at load time."""

    variant: TtsVariant | None = None
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class InstalledModel:
    """Model whose weights are present in the local models root."""

    name: str
    kind: ModelKind
    backend: str
    local_path: Path
    revision: str | None = None
    variant: TtsVariant | None = None
