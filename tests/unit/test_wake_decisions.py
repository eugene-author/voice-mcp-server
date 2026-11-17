"""The wake controller's pure decide(): the decision table, windows, timezones, backoff."""

from dataclasses import replace
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.domain.stt import SttBackendName, SttCapabilities, SttRuntime
from app.domain.workers import (
    CapabilitySnapshot,
    ControllerObservation,
    Decision,
    JobDemand,
    Schedule,
    ScheduleWindow,
    TargetState,
    WakeAction,
    WakePolicy,
    WorkerRecord,
    WorkerStatus,
    backoff_delay,
    decide,
)

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
POLICY = WakePolicy(
    idle_timeout=timedelta(seconds=300),
    min_uptime=timedelta(seconds=300),
    start_timeout=timedelta(seconds=900),
    drain_timeout=timedelta(seconds=1800),
    drain_confirm_margin=timedelta(seconds=60),
    lease_timeout=timedelta(seconds=120),
)
SNAPSHOT = CapabilitySnapshot(
    profile="linux-cuda",
    stt_runtime=SttRuntime(SttBackendName.CTRANSLATE2, "cuda", "float16"),
    stt_capabilities=SttCapabilities(True, True, True, True, True, True),
    tts_backend=None,
    tts_device=None,
    tts_capabilities=None,
    denoise_device=None,
    installed=frozenset(),
)
IDLE = JobDemand(runnable=0, running=0)
QUEUED = JobDemand(runnable=2, running=0)
RUNNING = JobDemand(runnable=0, running=1)


def worker(
    status: WorkerStatus = WorkerStatus.READY,
    *,
    started_ago: timedelta = 60 * MINUTE,
    beat_ago: timedelta = timedelta(seconds=5),
    active_ago: timedelta | None = 30 * MINUTE,
    drain_requested_ago: timedelta | None = None,
) -> WorkerRecord:
    """Return a registry row described relative to NOW."""
    return WorkerRecord(
        id="gpu:1",
        status=status,
        capabilities=SNAPSHOT,
        started_at=NOW - started_ago,
        heartbeat_at=NOW - beat_ago,
        last_activity_at=NOW - active_ago if active_ago is not None else None,
        drain_requested_at=NOW - drain_requested_ago if drain_requested_ago is not None else None,
    )


def observe(
    target: TargetState = TargetState.RUNNING,
    demand: JobDemand = IDLE,
    row: WorkerRecord | None = None,
    *,
    in_window: bool = False,
    started_ago: timedelta | None = None,
    running_since_ago: timedelta | None = None,
    backoff_for: timedelta | None = None,
) -> ControllerObservation:
    """Return an observation at NOW."""
    return ControllerObservation(
        now=NOW,
        demand=demand,
        target=target,
        worker=row,
        in_window=in_window,
        start_requested_at=NOW - started_ago if started_ago is not None else None,
        target_running_since=NOW - running_since_ago if running_since_ago is not None else None,
        backoff_until=NOW + backoff_for if backoff_for is not None else None,
    )


ROWS: list[tuple[str, ControllerObservation, WakeAction]] = [
    ("1 runnable jobs, target stopped", observe(TargetState.STOPPED, QUEUED, worker(WorkerStatus.STOPPED)), "start"),
    (
        "1 runnable jobs, start backoff pending",
        observe(TargetState.STOPPED, QUEUED, worker(WorkerStatus.STOPPED), backoff_for=MINUTE),
        "none",
    ),
    (
        "2 window within lead, target stopped",
        observe(TargetState.STOPPED, IDLE, worker(WorkerStatus.STOPPED), in_window=True),
        "start",
    ),
    ("3 worker busy", observe(row=worker(WorkerStatus.BUSY, active_ago=60 * MINUTE)), "none"),
    ("3 running job with fresh heartbeat", observe(demand=RUNNING, row=worker(active_ago=60 * MINUTE)), "none"),
    ("4 ready and idle", observe(row=worker()), "request_drain"),
    ("5 worker stopped, target running, no demand", observe(row=worker(WorkerStatus.STOPPED)), "stop"),
    (
        "6 draining and new demand",
        observe(demand=QUEUED, row=worker(WorkerStatus.DRAINING, drain_requested_ago=MINUTE)),
        "cancel_drain",
    ),
    (
        "6 drain requested, not yet seen, new demand",
        observe(demand=QUEUED, row=worker(drain_requested_ago=MINUTE)),
        "cancel_drain",
    ),
    (
        "6 signal drain cannot be cancelled",
        observe(demand=QUEUED, row=worker(WorkerStatus.DRAINING)),
        "none",
    ),
    (
        "5 not applied to the previous run's stopped row while a start is pending",
        observe(row=worker(WorkerStatus.STOPPED), started_ago=timedelta(seconds=5)),
        "none",
    ),
    ("7 stopping and new demand", observe(demand=QUEUED, row=worker(WorkerStatus.STOPPING)), "none"),
    ("7 stopped and new demand", observe(demand=QUEUED, row=worker(WorkerStatus.STOPPED)), "none"),
    (
        "8 draining past timeout plus margin",
        observe(row=worker(WorkerStatus.DRAINING, drain_requested_ago=timedelta(seconds=1861))),
        "force_stop",
    ),
    (
        "8 draining within timeout plus margin",
        observe(row=worker(WorkerStatus.DRAINING, drain_requested_ago=timedelta(seconds=1859))),
        "none",
    ),
    (
        "9 started but not ready in time",
        observe(
            TargetState.RUNNING, QUEUED, worker(WorkerStatus.STOPPED, beat_ago=2 * 60 * MINUTE), started_ago=15 * MINUTE
        ),
        "force_stop",
    ),
    (
        "9 still starting within the timeout",
        observe(TargetState.STARTING, QUEUED, None, started_ago=14 * MINUTE),
        "none",
    ),
    (
        "10 running, heartbeat lost longer than start timeout",
        observe(row=worker(beat_ago=16 * MINUTE)),
        "force_stop",
    ),
    (
        "10 running, never registered, seen running too long",
        observe(row=None, running_since_ago=16 * MINUTE),
        "force_stop",
    ),
    ("10 heartbeat lost only briefly", observe(row=worker(beat_ago=5 * MINUTE)), "none"),
    ("11 ready inside a window", observe(row=worker(), in_window=True), "none"),
    ("12 within min uptime", observe(row=worker(started_ago=4 * MINUTE, active_ago=None)), "none"),
    ("12 recent activity", observe(row=worker(active_ago=4 * MINUTE)), "none"),
    ("13 target starting", observe(TargetState.STARTING, QUEUED, None), "none"),
    ("13 target stopping", observe(TargetState.STOPPING, QUEUED, worker(WorkerStatus.STOPPED)), "none"),
    ("14 target unknown", observe(TargetState.UNKNOWN, QUEUED, None), "none"),
    ("stopped and no demand", observe(TargetState.STOPPED, IDLE, worker(WorkerStatus.STOPPED)), "none"),
    ("never registered, target stopped", observe(TargetState.STOPPED, IDLE, None), "start"),
]


@pytest.mark.parametrize(("observation", "action"), [(o, a) for _, o, a in ROWS], ids=[name for name, _, _ in ROWS])
def test_behavior_table(observation: ControllerObservation, action: str) -> None:
    """Return the action the behavior table specifies, with a reason."""
    decision = decide(observation, POLICY)
    assert decision.action is WakeAction(action)
    assert decision.reason


def test_failed_start_is_flagged_for_backoff() -> None:
    """Mark the force stop of a start that did not become ready, and only that one."""
    failed = decide(observe(TargetState.RUNNING, QUEUED, None, started_ago=15 * MINUTE), POLICY)
    lost = decide(observe(row=worker(beat_ago=16 * MINUTE)), POLICY)
    assert failed == Decision(WakeAction.FORCE_STOP, failed.reason, start_failed=True)
    assert not lost.start_failed


def test_ready_worker_clears_a_pending_start() -> None:
    """Keep a start whose worker became ready running, however long ago it was requested."""
    decision = decide(observe(row=worker(active_ago=MINUTE), started_ago=20 * MINUTE), POLICY)
    assert decision.action is WakeAction.NONE


@pytest.mark.parametrize(("failures", "seconds"), [(0, 0), (1, 30), (2, 60), (3, 120), (4, 300), (5, 300), (9, 300)])
def test_backoff_steps_and_repeats_the_last(failures: int, seconds: float) -> None:
    """Step through start_backoff_seconds per consecutive failure and stay on the last step."""
    assert backoff_delay(failures, (30, 60, 120, 300)) == timedelta(seconds=seconds)


ZAGREB = ZoneInfo("Europe/Zagreb")
WEEKDAY_MORNING = ScheduleWindow(
    days=frozenset(range(5)), start=time(8, 0), end=time(10, 0), lead=timedelta(minutes=10)
)


@pytest.mark.parametrize(
    ("local", "inside"),
    [
        (datetime(2026, 11, 2, 7, 49, tzinfo=ZAGREB), False),
        (datetime(2026, 11, 2, 7, 50, tzinfo=ZAGREB), True),
        (datetime(2026, 11, 2, 9, 59, 59, tzinfo=ZAGREB), True),
        (datetime(2026, 11, 2, 10, 0, tzinfo=ZAGREB), False),
        (datetime(2026, 11, 7, 8, 30, tzinfo=ZAGREB), False),
    ],
    ids=["before lead", "lead starts", "last second", "end is exclusive", "saturday"],
)
def test_window_edges_with_lead(local: datetime, inside: bool) -> None:
    """Contain [start - lead, end) on listed weekdays only."""
    schedule = Schedule(zone=ZAGREB, windows=(WEEKDAY_MORNING,))
    assert (schedule.active_window(local.astimezone(UTC)) is not None) is inside


def test_windows_follow_the_schedule_timezone() -> None:
    """Evaluate 08:00 Europe/Zagreb as 07:00 UTC in winter and 06:00 UTC in summer."""
    schedule = Schedule(zone=ZAGREB, windows=(replace(WEEKDAY_MORNING, lead=timedelta()),))
    assert schedule.active_window(datetime(2026, 11, 2, 7, 0, tzinfo=UTC)) is not None
    assert schedule.active_window(datetime(2026, 11, 2, 6, 59, tzinfo=UTC)) is None
    assert schedule.active_window(datetime(2026, 7, 6, 6, 0, tzinfo=UTC)) is not None


def test_lead_time_crosses_midnight_into_the_previous_day() -> None:
    """Start a Monday 00:05 window with a 10-minute lead at Sunday 23:55."""
    window = ScheduleWindow(days=frozenset({0}), start=time(0, 5), end=time(1, 0), lead=timedelta(minutes=10))
    schedule = Schedule(zone=UTC, windows=(window,))
    assert schedule.active_window(datetime(2026, 11, 1, 23, 55, tzinfo=UTC)) is window
    assert schedule.active_window(datetime(2026, 11, 1, 23, 54, tzinfo=UTC)) is None


def test_next_window_start_includes_the_lead() -> None:
    """Report the next start - lead after now, skipping the weekend."""
    schedule = Schedule(zone=ZAGREB, windows=(WEEKDAY_MORNING,))
    friday_evening = datetime(2026, 11, 6, 18, 0, tzinfo=UTC)
    assert schedule.next_start(friday_evening) == datetime(2026, 11, 9, 7, 50, tzinfo=ZAGREB)
    assert Schedule(zone=UTC).next_start(friday_evening) is None
