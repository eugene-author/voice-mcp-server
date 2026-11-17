"""The wake controller loop and the gpu operator over mock target, unit of work, clock and lock."""

import asyncio
from collections.abc import Iterator
from datetime import UTC, timedelta
from pathlib import Path

import pytest

from app import main
from app.domain.errors import ConfigurationError, InvalidRequestError
from app.domain.jobs import TranscribeCommand
from app.domain.workers import Schedule, TargetState, WakeAction, WakePolicy, WorkerStatus
from app.services.health_check_service import HealthCheckService, HealthReport
from app.services.runtime_catalog import RegistryRuntimeCatalog
from app.workers.gpu_operator import GpuOperator
from app.workers.reconciliation import AssetReconciler
from app.workers.wake_controller import CRASHED_START, REFUSED_START, WakeController
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness, host_profile, publish_worker
from tests.mocks.compute import MockComputeTarget
from tests.mocks.queue import MockWorkerLock
from tests.waiting import eventually

POLICY = WakePolicy(
    idle_timeout=timedelta(seconds=300),
    min_uptime=timedelta(seconds=300),
    start_timeout=timedelta(seconds=900),
    drain_timeout=timedelta(seconds=1800),
    drain_confirm_margin=timedelta(seconds=60),
    lease_timeout=timedelta(seconds=120),
)


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Return a harness whose jobs feed the controller's demand."""
    harness = build_harness(tmp_path)
    yield harness
    harness.executor.shutdown()


def _controller(
    harness: Harness,
    target: MockComputeTarget,
    *,
    dry_run: bool = False,
    lock: MockWorkerLock | None = None,
    tick_seconds: float = 3600,
    submitted: asyncio.Event | None = None,
) -> WakeController:
    """Return a controller over the harness database and clock."""
    return WakeController(
        uow=harness.uow,
        target=target,
        lock=lock or MockWorkerLock(),
        clock=harness.clock,
        schedule=Schedule(zone=UTC),
        policy=POLICY,
        backoff_steps=(30, 60),
        tick_seconds=tick_seconds,
        dry_run=dry_run,
        submitted=submitted or asyncio.Event(),
    )


async def _queue_job(harness: Harness) -> None:
    """Submit one transcription job."""
    await harness.jobs.submit(TranscribeCommand(input_path=write_tone(harness.inputs / "a.wav")), owner=OWNER)


async def _publish(harness: Harness, status: WorkerStatus, *, started_ago: float = 3600) -> None:
    """Register a fresh worker row started the given seconds ago and idle since then."""
    now = harness.clock.now()
    record = await publish_worker(
        harness.uow, harness.local_runtime.snapshot(), at=now - timedelta(seconds=started_ago), status=status
    )
    async with harness.uow() as uow:
        await uow.workers.beat(record.id, status=status, now=now, last_activity_at=None)


async def test_queued_job_starts_the_stopped_target(harness: Harness) -> None:
    """Start the target for runnable work and remember the start."""
    target = MockComputeTarget(TargetState.STOPPED)
    controller = _controller(harness, target)
    await _publish(harness, WorkerStatus.STOPPED)
    await _queue_job(harness)
    decision = await controller.tick()
    assert decision.action is WakeAction.START
    assert target.calls == ["start"]
    assert controller.memory.start_requested_at == harness.clock.now()
    assert controller.report()["last_decision"] == "start"


async def test_idle_ready_worker_is_asked_to_drain_then_stopped(harness: Harness) -> None:
    """Request a drain on the registry row, then stop the target once the worker reports stopped."""
    target = MockComputeTarget(TargetState.RUNNING)
    controller = _controller(harness, target)
    await _publish(harness, WorkerStatus.READY)
    assert (await controller.tick()).action is WakeAction.REQUEST_DRAIN
    assert harness.uow.db.workers["gpu-host:1"].drain_requested_at == harness.clock.now()
    assert target.calls == []
    async with harness.uow() as uow:
        await uow.workers.mark_stopped("gpu-host:1", now=harness.clock.now())
    assert (await controller.tick()).action is WakeAction.STOP
    assert target.calls == ["stop"]
    assert controller.memory.last_stop_at == harness.clock.now()


async def test_new_demand_cancels_a_requested_drain(harness: Harness) -> None:
    """Clear drain_requested_at when work arrives while the worker drains."""
    controller = _controller(harness, MockComputeTarget(TargetState.RUNNING))
    await _publish(harness, WorkerStatus.DRAINING)
    async with harness.uow() as uow:
        await uow.workers.request_drain("gpu-host:1", now=harness.clock.now())
    await _queue_job(harness)
    assert (await controller.tick()).action is WakeAction.CANCEL_DRAIN
    assert harness.uow.db.workers["gpu-host:1"].drain_requested_at is None


async def test_dry_run_decides_but_never_acts(harness: Harness) -> None:
    """Record and report decisions without starting, stopping or draining anything."""
    target = MockComputeTarget(TargetState.STOPPED)
    controller = _controller(harness, target, dry_run=True)
    await _queue_job(harness)
    assert (await controller.tick()).action is WakeAction.START
    target.state = TargetState.RUNNING
    await _publish(harness, WorkerStatus.READY)
    harness.uow.db.jobs.clear()
    assert (await controller.tick()).action is WakeAction.REQUEST_DRAIN
    assert target.calls == []
    assert harness.uow.db.workers["gpu-host:1"].drain_requested_at is None
    assert controller.memory.start_requested_at is None
    assert controller.report()["last_decision"] == "request_drain"


async def test_failed_start_force_stops_and_backs_off(harness: Harness) -> None:
    """Force-stop a start that did not become ready, step the backoff, and reset it after a successful start."""
    target = MockComputeTarget(TargetState.STOPPED)
    controller = _controller(harness, target)
    await _queue_job(harness)
    await controller.tick()
    target.state = TargetState.RUNNING
    harness.clock.advance(seconds=901)
    decision = await controller.tick()
    assert (decision.action, decision.start_failed) == (WakeAction.FORCE_STOP, True)
    assert controller.memory.start_failures == 1
    assert controller.memory.backoff_until == harness.clock.now() + timedelta(seconds=30)
    target.state = TargetState.STOPPED
    assert (await controller.tick()).action is WakeAction.NONE
    assert controller.report()["start_failed"] == "worker not ready within start_timeout_seconds"
    harness.clock.advance(seconds=31)
    assert (await controller.tick()).action is WakeAction.START
    target.state = TargetState.RUNNING
    await _publish(harness, WorkerStatus.READY, started_ago=0)
    await controller.tick()
    assert (controller.memory.start_failures, controller.memory.backoff_until) == (0, None)
    assert target.calls == ["start", "force_stop", "start"]


async def test_refused_start_backs_off_and_is_reported(harness: Harness) -> None:
    """Count a start the target refuses as a failed start, and step the backoff on each refusal."""
    target = MockComputeTarget(TargetState.STOPPED)
    target.fail_start = True
    controller = _controller(harness, target)
    await _queue_job(harness)
    assert (await controller.tick()).action is WakeAction.START
    assert controller.memory.start_failures == 1
    assert controller.memory.backoff_until == harness.clock.now() + timedelta(seconds=30)
    assert (await controller.tick()).action is WakeAction.NONE
    assert controller.report()["start_failed"] == REFUSED_START
    harness.clock.advance(seconds=31)
    assert (await controller.tick()).action is WakeAction.START
    assert controller.memory.start_failures == 2
    assert controller.memory.backoff_until == harness.clock.now() + timedelta(seconds=60)
    assert target.calls == ["start", "start"]


async def test_target_that_stops_before_the_worker_is_ready_backs_off(harness: Harness) -> None:
    """Count a target that exits during a start as a failed start, and reset once a worker is ready."""
    target = MockComputeTarget(TargetState.STOPPED)
    controller = _controller(harness, target)
    await _queue_job(harness)
    await controller.tick()
    target.state = TargetState.STOPPED
    decision = await controller.tick()
    assert (decision.action, decision.reason) == (
        WakeAction.NONE,
        "runnable jobs are waiting, but a start backoff is pending",
    )
    assert controller.memory.start_failures == 1
    assert controller.memory.start_requested_at is None
    assert controller.report()["start_failed"] == CRASHED_START
    harness.clock.advance(seconds=31)
    assert (await controller.tick()).action is WakeAction.START
    target.state = TargetState.STOPPED
    await controller.tick()
    assert controller.memory.start_failures == 2
    assert controller.memory.backoff_until == harness.clock.now() + timedelta(seconds=60)
    harness.clock.advance(seconds=61)
    await controller.tick()
    target.state = TargetState.RUNNING
    await _publish(harness, WorkerStatus.READY, started_ago=0)
    await controller.tick()
    assert (controller.memory.start_failures, controller.memory.start_failed) == (0, None)
    assert target.calls == ["start", "start", "start"]


async def test_stale_ready_row_does_not_hide_a_target_that_stopped(harness: Harness) -> None:
    """Keep the backoff when a dead worker's row still looks ready and fresh while the target is stopped."""
    target = MockComputeTarget(TargetState.STOPPED)
    controller = _controller(harness, target)
    await _queue_job(harness)
    await controller.tick()
    await _publish(harness, WorkerStatus.READY, started_ago=0)
    target.state = TargetState.STOPPED
    await controller.tick()
    assert controller.memory.start_failures == 1
    await controller.tick()
    assert controller.memory.start_failures == 1
    assert target.calls == ["start"]


async def test_failed_start_backoff_counts_from_the_end_of_the_forced_stop(harness: Harness) -> None:
    """Give a failed start its whole backoff even when stopping the target takes time."""
    target = MockComputeTarget(TargetState.STOPPED)
    target.on_stop = lambda: harness.clock.advance(seconds=12)
    controller = _controller(harness, target)
    await _queue_job(harness)
    await controller.tick()
    target.state = TargetState.RUNNING
    harness.clock.advance(seconds=901)
    await controller.tick()
    assert controller.memory.backoff_until == harness.clock.now() + timedelta(seconds=30)


async def test_failed_stop_is_not_counted_as_a_failed_start(harness: Harness) -> None:
    """Leave the backoff alone when an action other than a start fails."""
    target = MockComputeTarget(TargetState.RUNNING)

    def refuse() -> None:
        """Fail the stop call."""
        raise ConnectionError("platform refused the stop")

    target.on_stop = refuse
    controller = _controller(harness, target)
    await _publish(harness, WorkerStatus.STOPPED)
    assert (await controller.tick()).action is WakeAction.STOP
    assert (controller.memory.start_failures, controller.memory.backoff_until) == (0, None)


async def test_unknown_target_state_does_nothing_and_is_reported(harness: Harness) -> None:
    """Take no action when the platform cannot be asked, and say so in the report."""
    target = MockComputeTarget(TargetState.STOPPED)
    target.fail_status = True
    controller = _controller(harness, target)
    await _queue_job(harness)
    assert (await controller.tick()).action is WakeAction.NONE
    report = controller.report()
    assert (report["target_state"], report["reason"]) == ("unknown", "compute target state is unknown")
    assert target.calls == []


async def test_submission_triggers_an_immediate_tick(harness: Harness) -> None:
    """Start the target right after a submission instead of waiting for the next tick."""
    submitted = asyncio.Event()
    harness.jobs._submitted = submitted
    target = MockComputeTarget(TargetState.STOPPED)
    await _publish(harness, WorkerStatus.STOPPED)
    controller = _controller(harness, target, submitted=submitted, tick_seconds=3600)
    loop = asyncio.create_task(controller.run())
    try:
        await eventually(lambda: controller.report()["last_decision"] == "none")
        await _queue_job(harness)
        await eventually(lambda: target.calls == ["start"], within=2)
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)


async def test_controller_lock_keeps_a_second_instance_passive(harness: Harness) -> None:
    """Stay passive without the controller lock, and take over once it frees up."""
    lock = MockWorkerLock(available=False)
    target = MockComputeTarget(TargetState.STOPPED)
    await _queue_job(harness)
    controller = _controller(harness, target, lock=lock, tick_seconds=0.01)
    loop = asyncio.create_task(controller.run())
    try:
        await asyncio.sleep(0.05)
        assert target.calls == []
        assert controller.report()["controller"] == "standby"
        lock.available = True
        await eventually(lambda: target.calls == ["start"])
        assert lock.held
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
    assert not lock.held


async def test_quiet_week_keeps_the_last_worker_row_so_nothing_starts(tmp_path: Path) -> None:
    """Keep the newest stopped row through reconciliation: no phantom start, and a restarted api finds a snapshot."""
    harness = build_harness(tmp_path, registry=True)
    try:
        await publish_worker(
            harness.uow, harness.local_runtime.snapshot(), at=harness.clock.now(), status=WorkerStatus.STOPPED
        )
        harness.clock.advance(seconds=8 * 86400)
        reconciler = AssetReconciler(
            uow=harness.uow, store=harness.store, clock=harness.clock, retention_hours=24, interval_seconds=3600
        )
        await reconciler.reconcile()
        target = MockComputeTarget()
        decision = await _controller(harness, target).tick()
        assert decision.action is WakeAction.NONE
        assert target.calls == []
        restarted = RegistryRuntimeCatalog(uow=harness.uow, refresh_seconds=60)
        await restarted.refresh()
        assert restarted.snapshot().profile == harness.local_runtime.snapshot().profile
    finally:
        harness.executor.shutdown()


async def _health(harness: Harness, controller: WakeController) -> HealthReport:
    """Run health_check through the same gpu_worker provider the api role wires up."""
    service = HealthCheckService(
        host=host_profile(),
        runtime=harness.runtime,
        catalog=harness.model_host._catalog,
        checks={"storage": harness.store.verify},
        worker_state=lambda: "remote",
        data_root=harness.settings.data_root,
        workspace_root=harness.settings.workspace_root,
        gpu_worker=main._gpu_worker_status(harness.settings, uow=harness.uow, controller=controller),
    )
    return await service.check()


def _gpu_warnings(report: HealthReport) -> list[str]:
    """Return the report's warnings about the GPU worker."""
    return [warning for warning in report.details.get("warnings", []) if warning.startswith("GPU worker")]


async def test_idle_stopped_gpu_is_degraded_without_a_warning(harness: Harness) -> None:
    """Read degraded and stay quiet while the GPU is stopped and nothing asks for it."""
    controller = _controller(harness, MockComputeTarget())
    await controller.tick()
    report = await _health(harness, controller)
    assert report.status == "degraded"
    assert _gpu_warnings(report) == []


async def test_pending_start_warns_until_the_worker_is_ready(harness: Harness) -> None:
    """Warn that the GPU is starting through the whole cold start, and stop warning once the worker is ready."""
    target = MockComputeTarget()
    controller = _controller(harness, target)
    await _queue_job(harness)
    await controller.tick()
    report = await _health(harness, controller)
    assert report.status == "degraded"
    assert _gpu_warnings(report) == ["GPU worker is starting; 1 jobs waiting"]
    target.state = TargetState.RUNNING
    await controller.tick()
    assert _gpu_warnings(await _health(harness, controller)) == ["GPU worker is starting; 1 jobs waiting"]
    await _publish(harness, WorkerStatus.READY, started_ago=0)
    await controller.tick()
    assert _gpu_warnings(await _health(harness, controller)) == []


async def test_failed_start_warns_with_the_reason_and_next_attempt(harness: Harness) -> None:
    """Name the failure and the next attempt after a start that was forced to stop."""
    target = MockComputeTarget()
    controller = _controller(harness, target)
    await _queue_job(harness)
    await controller.tick()
    target.state = TargetState.RUNNING
    harness.clock.advance(seconds=901)
    await controller.tick()
    target.state = TargetState.STOPPED
    await controller.tick()
    report = await _health(harness, controller)
    next_attempt = controller.memory.backoff_until
    assert next_attempt is not None
    assert report.status == "degraded"
    assert _gpu_warnings(report) == [
        "GPU worker failed to start: worker not ready within start_timeout_seconds; "
        f"1 jobs waiting; next attempt at {next_attempt.isoformat()}"
    ]


async def test_refused_start_warns_with_the_reason_and_next_attempt(harness: Harness) -> None:
    """Report a refused start as a failure, not as a stopped GPU."""
    target = MockComputeTarget()
    target.fail_start = True
    controller = _controller(harness, target)
    await _queue_job(harness)
    await controller.tick()
    await controller.tick()
    next_attempt = controller.memory.backoff_until
    assert next_attempt is not None
    assert _gpu_warnings(await _health(harness, controller)) == [
        f"GPU worker failed to start: {REFUSED_START}; 1 jobs waiting; next attempt at {next_attempt.isoformat()}"
    ]


async def test_stopped_gpu_with_jobs_warns_even_with_no_start_in_progress(harness: Harness) -> None:
    """Warn in a dry run, where jobs wait on a stopped GPU with neither a pending nor a failed start."""
    controller = _controller(harness, MockComputeTarget(), dry_run=True)
    await _queue_job(harness)
    await controller.tick()
    assert _gpu_warnings(await _health(harness, controller)) == ["GPU worker is stopped; 1 jobs waiting"]


async def test_running_target_with_a_ready_worker_is_ok_without_a_warning(harness: Harness) -> None:
    """Read ok and stay quiet with the target running and its worker ready, even with jobs queued."""
    await _publish(harness, WorkerStatus.READY)
    await _queue_job(harness)
    controller = _controller(harness, MockComputeTarget(TargetState.RUNNING))
    await controller.tick()
    report = await _health(harness, controller)
    assert report.status == "ok"
    assert _gpu_warnings(report) == []


def _operator(harness: Harness, target: MockComputeTarget | None) -> GpuOperator:
    """Return the gpu CLI's operator over the harness."""
    return GpuOperator(
        uow=harness.uow,
        target=target,
        clock=harness.clock,
        schedule=Schedule(zone=UTC),
        policy=POLICY,
    )


async def test_operator_refuses_to_stop_a_running_job_without_force(harness: Harness) -> None:
    """Refuse gpu stop while a job runs, and stop with --force."""
    target = MockComputeTarget(TargetState.RUNNING)
    await _queue_job(harness)
    job_id = next(iter(harness.uow.db.jobs))
    async with harness.uow() as uow:
        await uow.jobs.claim(job_id, lease_owner="gpu-host:1", now=harness.clock.now())
    with pytest.raises(InvalidRequestError, match="--force"):
        await _operator(harness, target).stop(force=False)
    await _operator(harness, target).stop(force=True)
    assert target.calls == ["force_stop"]


async def test_operator_drains_the_active_worker_and_reports(harness: Harness) -> None:
    """Request a drain on the fresh worker row and report the observation and decision."""
    await _publish(harness, WorkerStatus.READY)
    operator = _operator(harness, MockComputeTarget(TargetState.RUNNING))
    assert await operator.drain() == "gpu-host:1"
    status = await operator.status()
    assert status["active_worker"]["drain_requested_at"] is not None
    assert status["decision"] == {"action": "none", "reason": "drain requested; waiting for the worker"}


async def test_operator_without_a_target_explains_the_setting(harness: Harness) -> None:
    """Explain that a target must be configured to start or stop, while status still works."""
    operator = _operator(harness, None)
    with pytest.raises(ConfigurationError, match="target is none"):
        await operator.start()
    with pytest.raises(InvalidRequestError, match="no GPU worker"):
        await operator.drain()
    assert (await operator.status())["target_state"] == "unknown"
