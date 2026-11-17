"""Removes unreferenced stored assets after the retention window, and old stopped worker rows but the newest."""

import asyncio
import logging
from datetime import timedelta

from app.ports.clock import Clock
from app.ports.storage import AssetStore
from app.ports.unit_of_work import UnitOfWorkFactory

logger = logging.getLogger(__name__)
LOOKUP_BATCH = 500
STOPPED_WORKER_RETENTION = timedelta(days=7)


class AssetReconciler:
    """Deletes orphans left by uploads whose database commit failed, after a retention window."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkFactory,
        store: AssetStore,
        clock: Clock,
        retention_hours: float,
        interval_seconds: float,
    ) -> None:
        """Store injected collaborators and timings."""
        self._uow = uow
        self._store = store
        self._clock = clock
        self._retention = timedelta(hours=retention_hours)
        self._interval = interval_seconds

    async def reconcile(self) -> int:
        """Delete unreferenced assets older than the retention window; return how many were removed."""
        candidates = await self._store.stale_objects(older_than=self._clock.now() - self._retention)
        removed = 0
        for start in range(0, len(candidates), LOOKUP_BATCH):
            batch = candidates[start : start + LOOKUP_BATCH]
            async with self._uow() as uow:
                known = await uow.assets.known_uris([location.uri for location in batch])
            for location in batch:
                if location.uri not in known:
                    await self._store.delete(location)
                    removed += 1
        if removed:
            logger.info("removed orphaned assets", extra={"count": removed})
        async with self._uow() as uow:
            workers = await uow.workers.delete_stopped(before=self._clock.now() - STOPPED_WORKER_RETENTION)
        if workers:
            logger.info("removed stopped worker registry rows", extra={"count": workers})
        return removed

    async def run(self) -> None:
        """Reconcile periodically until cancelled; this loop is a task boundary that logs and survives failures."""
        while True:
            try:
                await self.reconcile()
            except Exception:
                logger.exception("asset reconciliation failed")
            await asyncio.sleep(self._interval)
