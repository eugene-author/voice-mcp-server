"""Asset keys, locations and roles, built only from server-generated components."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.domain.caller import Caller
from app.domain.errors import InvalidRequestError
from app.domain.ids import ClaimToken, JobId, VoiceId

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
USERS_ROOT = "users"


class StorageBackend(StrEnum):
    """Durable asset storage backend."""

    LOCAL = "local"
    S3 = "s3"


class AssetRole(StrEnum):
    """Purpose of a stored asset."""

    INPUT = "input"
    WORKING = "working"
    OUTPUT = "output"
    VOICE_REFERENCE = "voice_reference"
    MANIFEST = "manifest"


PUBLIC_ROLES = frozenset({AssetRole.OUTPUT, AssetRole.MANIFEST})
"""Roles whose assets appear in the public files map."""


@dataclass(frozen=True, slots=True)
class AssetKey:
    """Relative storage key, identical for every backend."""

    value: str

    def __post_init__(self) -> None:
        """Reject keys that could escape the storage root."""
        parts = self.value.split("/")
        if self.value.startswith("/") or any(not _SAFE_COMPONENT.match(part) for part in parts):
            raise InvalidRequestError("invalid asset key")


@dataclass(frozen=True, slots=True)
class AssetLocation:
    """Durable location of an asset: an absolute local path or an s3:// URI."""

    backend: StorageBackend
    uri: str


@dataclass(frozen=True, slots=True)
class StoredAsset:
    """Result of publishing a finalized file."""

    location: AssetLocation
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PublishedAsset:
    """Published asset ready to be recorded in media_assets."""

    name: str
    """Key in the public files map, such as txt, json or wav."""

    role: AssetRole
    key: AssetKey
    stored: StoredAsset
    content_type: str
    duration_seconds: float | None = None
    sample_rate: int | None = None


INPUT_LOCATION_FIELD = "input_location"
"""Job payload field holding the uploaded input of storage.input_transfer=ingest."""


@dataclass(frozen=True, slots=True)
class UploadedInput:
    """An input copied into the asset store at submission; with ingest it is the job's preserved original."""

    key: AssetKey
    stored: StoredAsset
    content_type: str

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON stored in the job payload."""
        location = self.stored.location
        return {
            "key": self.key.value,
            "backend": str(location.backend),
            "uri": location.uri,
            "sha256": self.stored.sha256,
            "size_bytes": self.stored.size_bytes,
            "content_type": self.content_type,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> UploadedInput | None:
        """Return the job's uploaded input, or None for a job that reads a shared path."""
        data = payload.get(INPUT_LOCATION_FIELD)
        if not data:
            return None
        return cls(
            key=AssetKey(data["key"]),
            stored=StoredAsset(
                location=AssetLocation(StorageBackend(data["backend"]), data["uri"]),
                sha256=data["sha256"],
                size_bytes=int(data["size_bytes"]),
            ),
            content_type=data["content_type"],
        )

    @property
    def suffix(self) -> str:
        """Return the original file's extension with its dot, from the key."""
        name = self.key.value.rsplit("/", 1)[-1]
        return name[name.rfind(".") :] if "." in name else ""

    def as_original(self) -> PublishedAsset:
        """Return the asset row that records this object as the job's preserved original."""
        return PublishedAsset(
            name="original", role=AssetRole.INPUT, key=self.key, stored=self.stored, content_type=self.content_type
        )


def _component(value: str) -> str:
    """Return the value if it is a safe single path component, otherwise raise."""
    if not _SAFE_COMPONENT.match(value) or value in {".", ".."}:
        raise InvalidRequestError("invalid asset key component")
    return value


def _claim_scope(claim_token: ClaimToken | None) -> str:
    """Return the key segment that isolates one claim's uploads from every other claim's."""
    return f"{claim_token}/" if claim_token is not None else ""


def job_output_key(owner: Caller, job_id: JobId, name: str, *, claim_token: ClaimToken | None = None) -> AssetKey:
    """Return the key of a job output file; only a batch manifest, which no claim owns, leaves claim_token out."""
    return AssetKey(
        f"{USERS_ROOT}/{owner.owner_key}/jobs/{job_id}/outputs/{_claim_scope(claim_token)}{_component(name)}"
    )


def job_input_key(owner: Caller, job_id: JobId, extension: str, *, claim_token: ClaimToken | None = None) -> AssetKey:
    """Return the key of a job's original; an input uploaded at submission has no claim and is not rewritten."""
    original = f"original.{_component(extension.lower())}"
    return AssetKey(f"{USERS_ROOT}/{owner.owner_key}/jobs/{job_id}/inputs/{_claim_scope(claim_token)}{original}")


def voice_reference_key(
    owner: Caller, voice_id: VoiceId, version: int, extension: str, *, claim_token: ClaimToken
) -> AssetKey:
    """Return the key of a voice version's reference recording, private to the claim that uploads it."""
    voice = f"{USERS_ROOT}/{owner.owner_key}/voices/{_component(voice_id)}/v{version}"
    return AssetKey(f"{voice}/{claim_token}/reference.{_component(extension.lower())}")
