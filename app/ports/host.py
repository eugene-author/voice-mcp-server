"""Host probing port."""

from typing import Protocol

from app.domain.host import HostProfile


class HostProbe(Protocol):
    """Inspects hardware and installed extras."""

    def probe(self) -> HostProfile:
        """Return the host profile."""
        ...
