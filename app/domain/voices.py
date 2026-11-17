"""Cloned voices, versions, consent and runtime portability rules."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from app.domain.caller import Caller
from app.domain.errors import InvalidRequestError
from app.domain.ids import VoiceId, VoiceVersionId
from app.domain.tts import ConditioningMode, TtsCapabilities

MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 30.0
MAX_VOICE_NAME_LENGTH = 80


PRESET_VOICE_PREFIX = "preset:"
"""voice_id prefix that selects a preset speaker of the CustomVoice checkpoint instead of a cloned voice."""


class VoiceStatus(StrEnum):
    """Lifecycle state of a voice."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class VoiceKind(StrEnum):
    """Whether a voice is a built-in preset speaker or an owned clone."""

    PRESET = "preset"
    CLONED = "cloned"


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    """Affirmative consent captured at registration."""

    confirmed: bool
    confirmed_at: datetime
    source: str
    """How consent was given, such as the MCP tool name."""


@dataclass(frozen=True, slots=True)
class NewVoice:
    """Voice and its first version to persist in the pending state."""

    voice_id: VoiceId
    voice_version_id: VoiceVersionId
    owner: Caller
    name: str
    language: str | None
    reference_text: str | None
    conditioning_mode: ConditioningMode
    consent: ConsentRecord
    created_at: datetime


@dataclass(frozen=True, slots=True)
class VoiceVersion:
    """Immutable, runtime-portable reference data of one voice version."""

    id: VoiceVersionId
    voice_id: VoiceId
    version: int
    reference_uri: str | None
    """Durable location of the reference recording; None until registration completes."""

    reference_backend: str | None
    reference_sha256: str | None
    reference_text: str | None
    language: str | None
    conditioning_mode: ConditioningMode


@dataclass(frozen=True, slots=True)
class Voice:
    """Owned cloned voice with its current version."""

    id: VoiceId
    owner: Caller
    name: str
    status: VoiceStatus
    language: str | None
    created_at: datetime
    current_version: VoiceVersion | None


@dataclass(frozen=True, slots=True)
class VoiceSummary:
    """Voice entry returned by list_voices."""

    voice_id: str
    name: str
    kind: VoiceKind
    status: VoiceStatus
    language: str | None = None


def validate_voice_name(name: str) -> str:
    """Return the stripped name or raise if it is empty or too long."""
    stripped = name.strip()
    if not stripped or len(stripped) > MAX_VOICE_NAME_LENGTH:
        raise InvalidRequestError(f"voice name must be 1-{MAX_VOICE_NAME_LENGTH} characters")
    return stripped


def conditioning_mode_for(reference_text: str | None, capabilities: TtsCapabilities) -> ConditioningMode:
    """Return the conditioning mode for a registration, or raise if the runtime cannot clone that way."""
    if reference_text and reference_text.strip():
        if not capabilities.clone_icl:
            raise InvalidRequestError("the active TTS runtime does not support reference cloning")
        return ConditioningMode.ICL
    if not capabilities.clone_xvector_only:
        raise InvalidRequestError("reference_text is required on this host: the active TTS runtime supports ICL only")
    return ConditioningMode.XVECTOR


def ensure_usable(version: VoiceVersion, capabilities: TtsCapabilities) -> None:
    """Raise if the active runtime cannot condition on this voice version."""
    if version.conditioning_mode is ConditioningMode.XVECTOR and not capabilities.clone_xvector_only:
        raise InvalidRequestError(
            "this voice was registered without a transcript (x-vector mode), which the active TTS runtime "
            "does not support; re-register it with reference_text"
        )
    if version.conditioning_mode is ConditioningMode.ICL and not capabilities.clone_icl:
        raise InvalidRequestError("the active TTS runtime does not support reference cloning")
