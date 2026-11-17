"""Startup recovery and periodic polling for missed wake-ups, due retries and stale leases."""

import asyncio
import logging
from datetime import timedelta

from app.domain.jobs import JobStatus
from app.ports.clock import Clock
from app.ports.job_queue import JobQueue
from app.ports.unit_of_work import UnitOfWorkFactory
from app.services.batch_service import BatchService
from app.workers.job_transitions import apply_registration_rules

logger = logging.getLogger(__name__)
POLL_BATCH_SIZE = 100


class JobRecovery:
    """Re-enqueues work PostgreSQL knows about but the in-process queue may have missed."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkFactory,
        queue: JobQueue,
        batches: BatchService,
        clock: Clock,
        lease_timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> None:
        """Store injected collaborators and timings."""
        self._uow = uow
        self._queue = queue
        self._batches = batches
        self._clock = clock
        self._lease_timeout = timedelta(seconds=lease_timeout_seconds)
        self._poll_interval = poll_interval_seconds

    async def recover(self) -> None:
        """Interrupt stale jobs, requeue or fail them (failing exhausted registrations' voices), requeue due retries."""
        async with self._uow() as uow:
            now = self._clock.now()
            stale = await uow.jobs.interrupt_stale(heartbeat_before=now - self._lease_timeout)
            requeued, failed = await uow.jobs.resolve_interrupted(now=now)
            await apply_registration_rules(uow, failed, JobStatus.FAILED, now=now)
            due = await uow.jobs.requeue_due(now=now)
        if stale or failed or due:
            logger.info(
                "recovered jobs",
                extra={"interrupted": len(stale), "requeued": len(requeued), "failed": len(failed), "due": due},
            )
        await self._batches.reconcile_open_parents()
        await self.wake_queued()

    async def wake_queued(self) -> None:
        """Notify the worker of every queued job."""
        async with self._uow() as uow:
            job_ids = await uow.jobs.queued_ids(limit=POLL_BATCH_SIZE)
        for job_id in job_ids:
            self._queue.notify(job_id)

    async def poll_forever(self) -> None:
        """Recover periodically until cancelled; this loop is a task boundary that logs and survives failures."""
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self.recover()
            except Exception:
                logger.exception("recovery poll failed")
