"""Operator actions on the GPU worker for the `gpu` CLI: status, start, drain and stop."""

from datetime import datetime
from typing import Any

from app.domain.errors import ConfigurationError, InvalidRequestError
from app.domain.workers import Schedule, WakePolicy, WorkerRecord, decide
from app.ports.clock import Clock
from app.ports.compute import ComputeTarget
from app.ports.unit_of_work import UnitOfWorkFactory
from app.workers.wake_controller import ControllerMemory, observe

NO_TARGET = "gpu_worker.on_demand.target is none; configure docker or aws_ecs to start or stop the worker"


class GpuOperator:
    """Performs single operator actions; a running wake controller may reverse them on its next tick."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkFactory,
        target: ComputeTarget | None,
        clock: Clock,
        schedule: Schedule,
        policy: WakePolicy,
    ) -> None:
        """Store collaborators; target is None when no compute target is configured."""
        self._uow = uow
        self._target = target
        self._clock = clock
        self._schedule = schedule
        self._policy = policy

    async def status(self) -> dict[str, Any]:
        """Return the current observation, what the controller would decide now, and the active worker row."""
        now = self._clock.now()
        observation = await observe(
            uow=self._uow,
            target=self._target,
            schedule=self._schedule,
            policy=self._policy,
            now=now,
            memory=ControllerMemory(),
        )
        decision = decide(observation, self._policy)
        async with self._uow() as uow:
            active = await uow.workers.active(heartbeat_after=now - self._policy.lease_timeout)
        latest = observation.worker
        return {
            "target_state": str(observation.target),
            "demand": {"runnable": observation.demand.runnable, "running": observation.demand.running},
            "in_window": observation.in_window,
            "next_window_start": _iso(self._schedule.next_start(now)),
            "decision": {"action": str(decision.action), "reason": decision.reason},
            "active_worker": _worker(active),
            "latest_worker": _worker(latest),
        }

    async def start(self) -> None:
        """Start the compute target."""
        await self._require_target().start()

    async def drain(self) -> str:
        """Ask the active worker to drain; return its ID."""
        now = self._clock.now()
        async with self._uow() as uow:
            active = await uow.workers.active(heartbeat_after=now - self._policy.lease_timeout)
            if active is None:
                raise InvalidRequestError("no GPU worker is running")
            await uow.workers.request_drain(active.id, now=now)
        return active.id

    async def stop(self, *, force: bool) -> None:
        """Stop the compute target; refused while a job is running unless force is set."""
        target = self._require_target()
        if not force:
            now = self._clock.now()
            async with self._uow() as uow:
                demand = await uow.jobs.demand(now=now, heartbeat_before=now - self._policy.lease_timeout)
            if demand.running:
                raise InvalidRequestError(
                    f"{demand.running} job(s) running; drain first (gpu drain) or pass --force to stop now"
                )
        await target.stop(force=force)

    def _require_target(self) -> ComputeTarget:
        """Return the target or raise when none is configured."""
        if self._target is None:
            raise ConfigurationError(NO_TARGET)
        return self._target


def _worker(record: WorkerRecord | None) -> dict[str, Any] | None:
    """Return a worker row's operator-facing fields."""
    if record is None:
        return None
    return {
        "id": record.id,
        "status": str(record.status),
        "started_at": _iso(record.started_at),
        "heartbeat_at": _iso(record.heartbeat_at),
        "last_activity_at": _iso(record.last_activity_at),
        "drain_requested_at": _iso(record.drain_requested_at),
        "profile": record.capabilities.profile,
        "loaded_models": dict(record.capabilities.loaded),
    }


def _iso(moment: datetime | None) -> str | None:
    """Return an ISO timestamp or None."""
    return moment.isoformat() if moment else None
