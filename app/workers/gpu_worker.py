"""The worker role: registers in the worker registry, runs jobs, and drains and stops gracefully."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime

from app.domain.errors import ConfigurationError, VoiceMcpError
from app.domain.workers import CapabilitySnapshot, ModelRef, WorkerRecord, WorkerStatus
from app.ports.clock import Clock, MonotonicClock
from app.ports.locks import WorkerLock
from app.ports.runtime import RuntimeCatalog
from app.ports.unit_of_work import UnitOfWorkFactory
from app.services.model_host import ModelHost
from app.workers.job_worker import JobWorker
from app.workers.recovery import JobRecovery

logger = logging.getLogger(__name__)
STOP_UNLOAD_TIMEOUT_SECONDS = 30.0

type Loop = Callable[[], Coroutine[None, None, None]]


class GpuWorker:
    """Runs the job worker in its own process and follows the drain protocol."""

    def __init__(
        self,
        *,
        worker_id: str,
        version: str,
        lock: WorkerLock,
        worker: JobWorker,
        recovery: JobRecovery,
        model_host: ModelHost,
        runtime: RuntimeCatalog,
        record_preload_errors: Callable[[tuple[str, ...]], None],
        uow: UnitOfWorkFactory,
        clock: Clock,
        monotonic: MonotonicClock,
        heartbeat_seconds: float,
        poll_interval_seconds: float,
        drain_timeout_seconds: float,
        startup_recovery: bool,
        preload: tuple[ModelRef, ...] = (),
        loops: tuple[Loop, ...] = (),
    ) -> None:
        """Store collaborators and timings; loops run alongside the worker (listener, idle unload)."""
        self._id = worker_id
        self._version = version
        self._lock = lock
        self._worker = worker
        self._recovery = recovery
        self._model_host = model_host
        self._runtime = runtime
        self._record_preload_errors = record_preload_errors
        self._uow = uow
        self._clock = clock
        self._monotonic = monotonic
        self._heartbeat = heartbeat_seconds
        self._poll_interval = poll_interval_seconds
        self._drain_timeout = drain_timeout_seconds
        self._startup_recovery = startup_recovery
        self._preload = preload
        self._loops = loops
        self._status = WorkerStatus.STARTING
        self._started_at: datetime | None = None
        self._last_activity: datetime | None = None
        self._seen_activity = 0
        self._signalled = asyncio.Event()
        self._draining_since: float | None = None
        self._drain_by_signal = False
        self._abandoned_inference = False

    @property
    def status(self) -> WorkerStatus:
        """Return the status this process last recorded."""
        return self._status

    @property
    def abandoned_inference(self) -> bool:
        """Return whether a native call may still be running, so the process must exit without joining its thread."""
        return self._abandoned_inference

    def request_stop(self) -> None:
        """Begin a drain that cannot be cancelled, as SIGTERM and SIGINT do."""
        self._signalled.set()

    async def register_only(self) -> CapabilitySnapshot:
        """Publish this host's capability snapshot as a stopped worker without loading models, then return it."""
        now = self._clock.now()
        snapshot = self._runtime.snapshot()
        record = WorkerRecord(
            id=self._id,
            status=WorkerStatus.STOPPED,
            capabilities=snapshot,
            started_at=now,
            heartbeat_at=now,
            drained_at=now,
            version=self._version,
        )
        async with self._uow() as uow:
            await uow.workers.register(record)
        logger.info("registered worker capabilities", extra={"worker_id": self._id, "profile": snapshot.profile})
        return snapshot

    async def run(self) -> None:
        """Take the worker lock, register, preload, recover and process jobs until a drain completes."""
        if not await self._lock.acquire():
            raise ConfigurationError("another worker holds the worker lock; only one inference worker may run")
        try:
            await self._start()
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self._worker.run(), name="job-worker"),
                    group.create_task(self._poll(), name="wake-poll"),
                    *(group.create_task(loop(), name=f"worker-loop-{index}") for index, loop in enumerate(self._loops)),
                ]
                await self._control()
                for task in tasks:
                    task.cancel()
            await self._stop()
        finally:
            await asyncio.shield(self._lock.release())

    async def _start(self) -> None:
        """Register as starting, preload models, run startup recovery and become ready."""
        now = self._clock.now()
        self._started_at = now
        async with self._uow() as uow:
            await uow.workers.register(
                WorkerRecord(
                    id=self._id,
                    status=WorkerStatus.STARTING,
                    capabilities=self._runtime.snapshot(),
                    started_at=now,
                    heartbeat_at=now,
                    version=self._version,
                )
            )
        logger.info("gpu worker starting", extra={"worker_id": self._id})
        await self._preload_models()
        if self._startup_recovery:
            await self._recovery.recover()
        self._status = WorkerStatus.READY
        await self._beat()
        logger.info("gpu worker ready", extra={"worker_id": self._id})

    async def _preload_models(self) -> None:
        """Load the requested models; a failure is logged and reported in the snapshot, not fatal."""
        errors: list[str] = []
        for ref in self._preload:
            try:
                await self._model_host.preload(ref)
                logger.info("preloaded model", extra={"model": str(ref)})
            except VoiceMcpError as exc:
                logger.warning("model preload failed", extra={"model": str(ref), "error": exc.message})
                errors.append(f"{ref}: {exc.message}")
            except Exception as exc:
                logger.exception("model preload failed", extra={"model": str(ref)})
                errors.append(f"{ref}: {type(exc).__name__}")
        self._record_preload_errors(tuple(errors))

    async def _poll(self) -> None:
        """Wake the worker for queued jobs periodically, as a safety net for missed notifications."""
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._recovery.wake_queued()
            except Exception:
                logger.exception("wake poll failed")

    async def _control(self) -> None:
        """Heartbeat the registry and follow drain requests; return once the worker may stop."""
        while True:
            drain_requested = await self._beat()
            if self._draining_since is None:
                if self._signalled.is_set() or drain_requested:
                    self._begin_drain(by_signal=self._signalled.is_set())
                    continue
            elif not self._drain_by_signal and self._signalled.is_set():
                self._drain_by_signal = True
            elif not self._drain_by_signal and drain_requested is False:
                self._cancel_drain()
                continue
            if self._draining_since is not None:
                if not self._worker.busy:
                    return
                if self._monotonic.monotonic() - self._draining_since >= self._drain_timeout:
                    logger.warning("drain deadline reached; releasing the running job", extra={"worker_id": self._id})
                    self._abandoned_inference = await self._worker.release_current()
                    return
            await self._pause()

    def _begin_drain(self, *, by_signal: bool) -> None:
        """Stop claiming; the current job, if any, runs to completion."""
        self._worker.stop_claiming()
        self._draining_since = self._monotonic.monotonic()
        self._drain_by_signal = by_signal
        self._status = WorkerStatus.DRAINING
        logger.info("gpu worker draining", extra={"worker_id": self._id, "signal": by_signal})

    def _cancel_drain(self) -> None:
        """Resume claiming after the controller cleared its drain request."""
        self._worker.resume_claiming()
        self._draining_since = None
        self._status = WorkerStatus.BUSY if self._worker.busy else WorkerStatus.READY
        logger.info("gpu worker drain cancelled", extra={"worker_id": self._id})

    async def _pause(self) -> None:
        """Sleep until the next heartbeat, a signal not yet acted on, the drain deadline, or the drained job's end."""
        timeout = self._heartbeat
        waits: list[asyncio.Task[object]] = []
        if not self._drain_by_signal:
            waits.append(asyncio.create_task(self._signalled.wait()))
        if self._draining_since is not None:
            waits.append(asyncio.create_task(self._worker.wait_idle()))
            remaining = self._draining_since + self._drain_timeout - self._monotonic.monotonic()
            timeout = max(0.0, min(timeout, remaining))
        try:
            await asyncio.wait(waits, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for wait in waits:
                wait.cancel()
            await asyncio.gather(*waits, return_exceptions=True)

    async def _beat(self) -> bool | None:
        """Record status, activity and capabilities; return whether a drain is requested, None if unknown."""
        if self._draining_since is None and self._status in (WorkerStatus.READY, WorkerStatus.BUSY):
            self._status = WorkerStatus.BUSY if self._worker.busy else WorkerStatus.READY
        now = self._clock.now()
        if self._worker.activity != self._seen_activity or self._worker.busy:
            self._seen_activity = self._worker.activity
            self._last_activity = now
        try:
            async with self._uow() as uow:
                record = await uow.workers.beat(
                    self._id,
                    status=self._status,
                    now=now,
                    last_activity_at=self._last_activity,
                    capabilities=self._runtime.snapshot(),
                )
        except Exception:
            logger.exception("worker registry heartbeat failed", extra={"worker_id": self._id})
            return None
        if record is None:
            logger.warning("worker registry row disappeared", extra={"worker_id": self._id})
            return None
        return record.drain_requested_at is not None

    async def _stop(self) -> None:
        """Mark stopping, unload unless a native call was abandoned, release the worker lock, then record stopped."""
        self._status = WorkerStatus.STOPPING
        await self._beat()
        if not self._abandoned_inference:
            await self._unload()
        await self._lock.release()
        async with self._uow() as uow:
            await uow.workers.mark_stopped(self._id, now=self._clock.now())
        self._status = WorkerStatus.STOPPED
        logger.info("gpu worker stopped", extra={"worker_id": self._id})

    async def _unload(self) -> None:
        """Unload every model within the time limit; a timeout counts as abandoned, and no failure prevents stopping."""
        try:
            async with asyncio.timeout(STOP_UNLOAD_TIMEOUT_SECONDS):
                await self._model_host.unload_all()
        except TimeoutError:
            self._abandoned_inference = True
            logger.warning("unloading models at stop timed out", extra={"worker_id": self._id})
        except Exception:
            logger.exception("unloading models at stop failed", extra={"worker_id": self._id})
