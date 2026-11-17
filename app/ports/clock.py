"""Wall-clock and monotonic clock ports."""

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    """Source of timezone-aware wall-clock time."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        ...


class MonotonicClock(Protocol):
    """Source of monotonic time for intervals and TTLs."""

    def monotonic(self) -> float:
        """Return monotonic seconds."""
        ...
