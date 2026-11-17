"""System clock implementations."""

import time
from datetime import UTC, datetime


class SystemClock:
    """Wall clock returning timezone-aware UTC datetimes."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        return datetime.now(tz=UTC)


class SystemMonotonicClock:
    """Monotonic clock for intervals and TTLs."""

    def monotonic(self) -> float:
        """Return monotonic seconds."""
        return time.monotonic()
