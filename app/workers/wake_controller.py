"""Wake controller: a level-triggered loop in the api role that starts, drains and stops the GPU worker."""

import asyncio
import contextlib
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from app.domain.workers import (
    ControllerObservation,
    Decision,
    Schedule,
    TargetState,
    WakeAction,
    WakePolicy,
    backoff_delay,
    decide,
    worker_ready,
)
from app.ports.clock import Clock
from app.ports.compute import ComputeTarget
from app.ports.locks import WorkerLock
from app.ports.unit_of_work import UnitOfWorkFactory

logger = logging.getLogger(__name__)

CRASHED_START = "compute target stopped before the worker was ready"
REFUSED_START = "compute target refused the start"


@dataclass(frozen=True, slots=True)
class ControllerMemory:
    """What the controller remembers between ticks."""

    start_requested_at: datetime | None = None
    target_running_since: datetime | None = None
    start_failures: int = 0
    backoff_until: datetime | None = None
    last_start_at: datetime | None = None
    last_stop_at: datetime | None = None
    start_failed: str | None = None


async def observe(
    *,
    uow: UnitOfWorkFactory,
    target: ComputeTarget | None,
    schedule: Schedule,
    policy: WakePolicy,
    now: datetime,
    memory: ControllerMemory,
) -> ControllerObservation:
    """Read job demand, the newest worker row, the target state (unknown without one) and the schedule."""
    async with uow() as work:
        demand = await work.jobs.demand(now=now, heartbeat_before=now - policy.lease_timeout)
        worker = await work.workers.latest()
    state = TargetState.UNKNOWN
    if target is not None:
        try:
            state = await target.status()
        except Exception:
            logger.exception("compute target status failed")
    return ControllerObservation(
        now=now,
        demand=demand,
        target=state,
        worker=worker,
        in_window=schedule.active_window(now) is not None,
        start_requested_at=memory.start_requested_at,
        target_running_since=memory.target_running_since,
        backoff_until=memory.backoff_until,
    )


async def act(
    decision: Decision, observation: ControllerObservation, *, uow: UnitOfWorkFactory, target: ComputeTarget
) -> None:
    """Carry out one decision against the compute target or the worker registry."""
    worker = observation.worker
    match decision.action:
        case WakeAction.START:
            await target.start()
        case WakeAction.STOP | WakeAction.FORCE_STOP:
            await target.stop(force=decision.action is WakeAction.FORCE_STOP)
        case WakeAction.REQUEST_DRAIN if worker is not None:
            async with uow() as work:
                await work.workers.request_drain(worker.id, now=observation.now)
        case WakeAction.CANCEL_DRAIN if worker is not None:
            async with uow() as work:
                await work.workers.cancel_drain(worker.id)


class WakeController:
    """Observes, decides with the pure decide(), and acts every tick; a submission requests an immediate tick."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkFactory,
        target: ComputeTarget,
        lock: WorkerLock,
        clock: Clock,
        schedule: Schedule,
        policy: WakePolicy,
        backoff_steps: tuple[float, ...],
        tick_seconds: float,
        dry_run: bool,
        submitted: asyncio.Event,
    ) -> None:
        """Store collaborators; submitted is set by JobService after each committed submission."""
        self._uow = uow
        self._target = target
        self._lock = lock
        self._clock = clock
        self._schedule = schedule
        self._policy = policy
        self._backoff_steps = backoff_steps
        self._tick = tick_seconds
        self._dry_run = dry_run
        self._submitted = submitted
        self._memory = ControllerMemory()
        self._observation: ControllerObservation | None = None
        self._decision: Decision | None = None
        self._holds_lock = False

    @property
    def memory(self) -> ControllerMemory:
        """Return the controller's memory."""
        return self._memory

    async def run(self) -> None:
        """Tick until cancelled while holding the controller lock; without it, retry the lock every tick."""
        try:
            while True:
                if not self._holds_lock:
                    self._holds_lock = await self._try_lock()
                if self._holds_lock:
                    await self.tick()
                await self._wait()
        finally:
            if self._holds_lock:
                await asyncio.shield(self._lock.release())

    async def tick(self) -> Decision:
        """Observe, decide, remember and act once; this is a task boundary that logs and survives failures."""
        now = self._clock.now()
        try:
            observation = await observe(
                uow=self._uow,
                target=self._target,
                schedule=self._schedule,
                policy=self._policy,
                now=now,
                memory=self._memory,
            )
        except Exception:
            logger.exception("wake controller could not observe")
            return Decision(WakeAction.NONE, "observation failed")
        self._memory = _remember_observation(self._memory, observation, self._policy, self._backoff_steps)
        decision = decide(
            replace(
                observation,
                start_requested_at=self._memory.start_requested_at,
                backoff_until=self._memory.backoff_until,
            ),
            self._policy,
        )
        self._record(observation, decision)
        if decision.action is WakeAction.NONE or self._dry_run:
            return decision
        try:
            await act(decision, observation, uow=self._uow, target=self._target)
        except Exception:
            logger.exception("wake controller action failed", extra={"action": str(decision.action)})
            if decision.action is WakeAction.START:
                self._memory = _failed_start(self._memory, REFUSED_START, self._clock.now(), self._backoff_steps)
            return decision
        self._memory = _remember_action(self._memory, decision, self._clock.now(), self._backoff_steps)
        return decision

    def report(self) -> dict[str, Any]:
        """Return the latest observation, decision and memory for health_check."""
        observation, decision, memory = self._observation, self._decision, self._memory
        worker = observation.worker if observation else None
        next_start = self._schedule.next_start(self._clock.now())
        return {
            "controller": "active" if self._holds_lock else "standby",
            "target_state": str(observation.target) if observation else None,
            "worker_status": str(worker.status) if worker else None,
            "last_decision": str(decision.action) if decision else None,
            "reason": decision.reason if decision else None,
            "last_start_at": _iso(memory.last_start_at),
            "last_stop_at": _iso(memory.last_stop_at),
            "start_backoff": {"failures": memory.start_failures, "until": _iso(memory.backoff_until)},
            "next_window_start": _iso(next_start),
            "start_failed": memory.start_failed,
        }

    @property
    def gpu_stopped(self) -> bool:
        """Return whether the target was last seen stopped, with or without demand; health then reads degraded."""
        observation = self._observation
        return observation is not None and observation.target is TargetState.STOPPED

    @property
    def demand_warning(self) -> str | None:
        """Return a health warning while jobs or a window wait for a GPU worker that is not up, otherwise None."""
        observation, memory = self._observation, self._memory
        if observation is None or not observation.new_demand or worker_ready(observation, self._policy):
            return None
        waiting = f"{observation.demand.runnable} jobs waiting"
        if memory.start_requested_at is not None:
            return f"GPU worker is starting; {waiting}"
        if memory.start_failed is not None:
            next_attempt = _iso(memory.backoff_until)
            return f"GPU worker failed to start: {memory.start_failed}; {waiting}; next attempt at {next_attempt}"
        if observation.target is TargetState.STOPPED:
            return f"GPU worker is stopped; {waiting}"
        return None

    async def _try_lock(self) -> bool:
        """Try the controller lock; log once when another replica controls the GPU."""
        try:
            acquired = await self._lock.acquire()
        except Exception:
            logger.exception("wake controller could not take its lock")
            return False
        if acquired:
            logger.info("wake controller active", extra={"dry_run": self._dry_run})
        return acquired

    async def _wait(self) -> None:
        """Sleep one tick, waking early when a job is submitted."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._submitted.wait(), timeout=self._tick)
        self._submitted.clear()

    def _record(self, observation: ControllerObservation, decision: Decision) -> None:
        """Keep the latest observation and log each change of decision once."""
        previous = self._decision
        self._observation, self._decision = observation, decision
        if previous is None or (previous.action, previous.reason) != (decision.action, decision.reason):
            if observation.target is TargetState.UNKNOWN:
                logger.warning("compute target state is unknown")
            logger.info(
                "wake controller decision",
                extra={"action": str(decision.action), "reason": decision.reason, "dry_run": self._dry_run},
            )


def _failed_start(
    memory: ControllerMemory, reason: str, now: datetime, backoff_steps: tuple[float, ...]
) -> ControllerMemory:
    """Forget the pending start and hold the next one back for the following backoff step."""
    failures = memory.start_failures + 1
    return replace(
        memory,
        start_requested_at=None,
        start_failures=failures,
        backoff_until=now + backoff_delay(failures, backoff_steps),
        start_failed=reason,
    )


def _remember_observation(
    memory: ControllerMemory,
    observation: ControllerObservation,
    policy: WakePolicy,
    backoff_steps: tuple[float, ...],
) -> ControllerMemory:
    """Track when the target started running, fail a start whose target stopped, and clear one that became ready."""
    running = observation.target is TargetState.RUNNING
    since = (memory.target_running_since or observation.now) if running else None
    memory = replace(memory, target_running_since=since)
    if observation.target is TargetState.STOPPED:
        if memory.start_requested_at is None:
            return memory
        return _failed_start(memory, CRASHED_START, observation.now, backoff_steps)
    if worker_ready(observation, policy):
        return replace(memory, start_requested_at=None, start_failures=0, backoff_until=None, start_failed=None)
    return memory


def _remember_action(
    memory: ControllerMemory, decision: Decision, now: datetime, backoff_steps: tuple[float, ...]
) -> ControllerMemory:
    """Record a start, a stop, or a failed start that advances the backoff."""
    match decision.action:
        case WakeAction.START:
            return replace(memory, start_requested_at=now, last_start_at=now)
        case WakeAction.STOP | WakeAction.FORCE_STOP if decision.start_failed:
            return replace(_failed_start(memory, decision.reason, now, backoff_steps), last_stop_at=now)
        case WakeAction.STOP | WakeAction.FORCE_STOP:
            return replace(memory, start_requested_at=None, last_stop_at=now)
    return memory


def _iso(moment: datetime | None) -> str | None:
    """Return an ISO timestamp or None."""
    return moment.isoformat() if moment else None
