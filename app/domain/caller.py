"""Authenticated caller identity."""

import hashlib
from dataclasses import dataclass

LOCAL_ISSUER = "local"


@dataclass(frozen=True, slots=True)
class Caller:
    """Authenticated caller; ownership uses issuer and subject."""

    issuer: str
    subject: str
    client_id: str | None = None

    @property
    def owner_key(self) -> str:
        """Return a stable opaque key: 32 hex chars of SHA-256 over issuer, NUL and subject."""
        digest = hashlib.sha256(f"{self.issuer}\0{self.subject}".encode()).hexdigest()
        return digest[:32]
