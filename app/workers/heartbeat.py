"""Heartbeat that keeps the lease of the executing claim fresh and notices when the claim is lost."""

import asyncio
import logging
from collections.abc import Callable

from app.domain.jobs import ClaimedJob
from app.ports.clock import Clock
from app.ports.unit_of_work import UnitOfWorkFactory

logger = logging.getLogger(__name__)


class Heartbeat:
    """Refreshes heartbeat_at of the job the worker is executing."""

    def __init__(self, *, uow: UnitOfWorkFactory, clock: Clock, interval_seconds: float) -> None:
        """Store injected collaborators and the interval."""
        self._uow = uow
        self._clock = clock
        self._interval = interval_seconds

    async def beat(self, job: ClaimedJob) -> bool:
        """Refresh the claim's lease once; return whether the claim still holds the job."""
        async with self._uow() as uow:
            return await uow.jobs.heartbeat(job.id, claim_token=job.claim_token, now=self._clock.now())

    async def keep_alive(self, job: ClaimedJob, *, on_lost: Callable[[], None]) -> None:
        """Beat until cancelled; a database failure is logged and retried, and a rejection calls on_lost and ends."""
        while True:
            await asyncio.sleep(self._interval)
            try:
                held = await self.beat(job)
            except Exception:
                logger.exception("heartbeat failed", extra={"job_id": str(job.id)})
                continue
            if not held:
                logger.warning("heartbeat rejected; the claim was lost", extra={"job_id": str(job.id)})
                on_lost()
                return
