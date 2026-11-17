"""Controllable clocks."""

from datetime import UTC, datetime, timedelta


class MockClock:
    """Wall clock that only moves when advanced."""

    def __init__(self, start: datetime | None = None) -> None:
        """Start at a fixed UTC instant."""
        self._now = start or datetime(2026, 11, 1, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        """Return the current mock time."""
        return self._now

    def advance(self, *, seconds: float) -> None:
        """Move time forward."""
        self._now += timedelta(seconds=seconds)


class MockMonotonicClock:
    """Monotonic clock that only moves when advanced."""

    def __init__(self, start: float = 1000.0) -> None:
        """Start at a fixed value."""
        self._value = start

    def monotonic(self) -> float:
        """Return the current mock monotonic time."""
        return self._value

    def advance(self, *, seconds: float) -> None:
        """Move time forward."""
        self._value += seconds
