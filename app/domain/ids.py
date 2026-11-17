"""Identifier types and time-ordered UUID generation."""

import uuid
from typing import NewType

JobId = NewType("JobId", uuid.UUID)
VoiceId = NewType("VoiceId", str)
VoiceVersionId = NewType("VoiceVersionId", uuid.UUID)
AssetId = NewType("AssetId", uuid.UUID)
ClaimToken = NewType("ClaimToken", uuid.UUID)


def uuid7() -> uuid.UUID:
    """Return an RFC 9562 version 7 UUID, monotonic within the process."""
    return uuid.uuid7()


def new_job_id() -> JobId:
    """Return a new time-ordered job identifier."""
    return JobId(uuid7())


def new_voice_id() -> VoiceId:
    """Return a new opaque voice identifier."""
    return VoiceId(f"voice_{uuid7().hex}")


def new_voice_version_id() -> VoiceVersionId:
    """Return a new voice version identifier."""
    return VoiceVersionId(uuid7())


def new_claim_token() -> ClaimToken:
    """Return the identity of one successful job claim."""
    return ClaimToken(uuid7())


def new_asset_id() -> AssetId:
    """Return a new asset identifier."""
    return AssetId(uuid7())
