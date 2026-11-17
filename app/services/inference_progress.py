"""Estimated job progress while a model runs, from recent work and kept below the next real checkpoint."""

import asyncio
import contextlib
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.domain.errors import ClaimLostError
from app.domain.jobs import JobPhase, ProgressCallback
from app.ports.clock import MonotonicClock

logger = logging.getLogger(__name__)

STT_SECONDS_PER_AUDIO_SECOND = 1.0
TTS_SECONDS_PER_CHARACTER = 0.7
DEFAULT_INTERVAL_SECONDS = 2.0
MIN_EXPECTED_SECONDS = 0.1
CURVE_STEEPNESS = 2.0
LEARNING_WEIGHT = 0.5


def estimated_percentage(*, start: int, end: int, elapsed_seconds: float, expected_seconds: float) -> int:
    """Return a percentage that climbs from start, slows down over time and always stays below end."""
    share = 1 - math.exp(-CURVE_STEEPNESS * elapsed_seconds / expected_seconds)
    return max(start, min(end - 1, start + int((end - start) * share)))


class InferenceProgress:
    """Reports estimated progress during inference and learns each model's speed on this host from finished work."""

    def __init__(
        self,
        *,
        monotonic: MonotonicClock,
        seconds_per_unit: float,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        """Start every model from a rough speed; a unit is whatever the caller measures work in."""
        self._monotonic = monotonic
        self._initial_speed = seconds_per_unit
        self._speeds: dict[str, float] = {}
        self._interval = interval_seconds

    def seconds_per_unit(self, key: str) -> float:
        """Return the current speed of one model, or the starting speed for a model not seen yet."""
        return self._speeds.get(key, self._initial_speed)

    @asynccontextmanager
    async def track(
        self, progress: ProgressCallback, *, phase: JobPhase, start: int, end: int, units: float, key: str, learn: bool
    ) -> AsyncIterator[None]:
        """Report estimates during the block; with learn, finished work updates this model's speed."""
        started = self._monotonic.monotonic()
        speed = self.seconds_per_unit(key)
        expected = max(speed * units, MIN_EXPECTED_SECONDS)
        ticker = asyncio.create_task(self._tick(progress, phase, start, end, started, expected), name="progress")
        try:
            yield
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
        if learn and units > 0:
            observed = (self._monotonic.monotonic() - started) / units
            self._speeds[key] = speed + LEARNING_WEIGHT * (observed - speed)

    async def _tick(
        self, progress: ProgressCallback, phase: JobPhase, start: int, end: int, started: float, expected: float
    ) -> None:
        """Report each new percentage; this task boundary logs a failed write and stops once the claim is lost."""
        reported = start
        while True:
            await asyncio.sleep(self._interval)
            elapsed = self._monotonic.monotonic() - started
            percentage = estimated_percentage(start=start, end=end, elapsed_seconds=elapsed, expected_seconds=expected)
            if percentage == reported:
                continue
            try:
                await progress(phase, percentage)
            except ClaimLostError:
                return
            except Exception:
                logger.exception("could not record estimated progress")
                continue
            reported = percentage
