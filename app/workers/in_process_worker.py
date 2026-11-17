"""In-process worker of the all role: single-worker lock, recovery and the worker loop in one TaskGroup."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from enum import StrEnum

from app.ports.locks import WorkerLock
from app.workers.job_worker import JobWorker
from app.workers.recovery import JobRecovery

logger = logging.getLogger(__name__)


class InProcessWorkerState(StrEnum):
    """Whether this process runs inference."""

    STARTING = "starting"
    ACTIVE = "active"
    STANDBY = "standby"
    STOPPED = "stopped"


class InProcessWorker:
    """Owns background startup and shutdown of the worker and recovery polling."""

    def __init__(
        self,
        *,
        lock: WorkerLock,
        worker: JobWorker,
        recovery: JobRecovery,
        startup_recovery: bool,
        extra_loops: tuple[Callable[[], Coroutine[None, None, None]], ...] = (),
    ) -> None:
        """Store the background components."""
        self._lock = lock
        self._worker = worker
        self._recovery = recovery
        self._startup_recovery = startup_recovery
        self._extra_loops = extra_loops
        self._state = InProcessWorkerState.STARTING

    @property
    def state(self) -> InProcessWorkerState:
        """Return the current worker state."""
        return self._state

    async def run(self) -> None:
        """Take the lock, recover, then run the loops until cancelled; without the lock serve as standby."""
        if not await self._lock.acquire():
            self._state = InProcessWorkerState.STANDBY
            logger.warning("another process holds the worker lock; serving reads and submissions only")
            return
        try:
            self._state = InProcessWorkerState.ACTIVE
            logger.info("inference worker started")
            if self._startup_recovery:
                await self._recovery.recover()
            async with asyncio.TaskGroup() as group:
                group.create_task(self._worker.run(), name="job-worker")
                group.create_task(self._recovery.poll_forever(), name="job-recovery")
                for index, loop in enumerate(self._extra_loops):
                    group.create_task(loop(), name=f"background-{index}")
        finally:
            self._state = InProcessWorkerState.STOPPED
            await asyncio.shield(self._lock.release())
