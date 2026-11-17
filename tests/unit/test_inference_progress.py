"""Estimated progress during inference, the job events it writes and the running job it updates."""

import asyncio
from pathlib import Path

import pytest

from app.domain.errors import ClaimLostError
from app.domain.jobs import JobPhase, JobStatus, TranscribeCommand
from app.services.inference_progress import InferenceProgress, estimated_percentage
from tests.conftest import write_tone
from tests.harness import OWNER, build_harness
from tests.mocks.clock import MockMonotonicClock
from tests.mocks.stt import MockBlockingSttBackend
from tests.waiting import eventually

INTERVAL = 0.001
MODEL = "small"


@pytest.mark.parametrize(
    ("elapsed", "expected_percentage"),
    [(0, 30), (5, 64), (10, 77), (20, 83), (1000, 84)],
)
def test_estimate_climbs_slower_over_time_and_stays_below_the_end(elapsed: float, expected_percentage: int) -> None:
    """Start at start, climb, and stay below end however long inference takes."""
    assert estimated_percentage(start=30, end=85, elapsed_seconds=elapsed, expected_seconds=10) == expected_percentage


def test_estimate_never_falls_below_start_on_a_tiny_span() -> None:
    """Stay at start when a chunk's share of the bar rounds to nothing."""
    assert estimated_percentage(start=40, end=40, elapsed_seconds=99, expected_seconds=1) == 40


async def _track(
    tracker: InferenceProgress,
    reports: list[tuple[JobPhase, int]],
    clock: MockMonotonicClock,
    seconds: float,
    *,
    key: str = MODEL,
    learn: bool = True,
) -> None:
    """Run one tracked block that spends the given seconds, and collect a report."""

    async def record(phase: JobPhase, percentage: int) -> None:
        """Collect one report."""
        reports.append((phase, percentage))

    async with tracker.track(record, phase=JobPhase.INFERENCING, start=30, end=85, units=10, key=key, learn=learn):
        clock.advance(seconds=seconds)
        await eventually(lambda: bool(reports))


async def test_tracking_reports_a_percentage_between_the_checkpoints() -> None:
    """Report the estimate while the block runs."""
    clock = MockMonotonicClock()
    tracker = InferenceProgress(monotonic=clock, seconds_per_unit=1.0, interval_seconds=INTERVAL)
    reports: list[tuple[JobPhase, int]] = []
    await _track(tracker, reports, clock, seconds=5)
    assert reports[0] == (JobPhase.INFERENCING, 64)
    assert len(set(reports)) == 1


async def test_finished_work_teaches_the_speed_of_this_host() -> None:
    """Move the estimate halfway to the speed just observed."""
    clock = MockMonotonicClock()
    tracker = InferenceProgress(monotonic=clock, seconds_per_unit=1.0, interval_seconds=INTERVAL)
    await _track(tracker, [], clock, seconds=30)
    assert tracker.seconds_per_unit(MODEL) == pytest.approx(2.0)


async def test_each_model_learns_its_own_speed() -> None:
    """Keep a slow model's speed away from every other model."""
    clock = MockMonotonicClock()
    tracker = InferenceProgress(monotonic=clock, seconds_per_unit=1.0, interval_seconds=INTERVAL)
    await _track(tracker, [], clock, seconds=30, key="large")
    assert tracker.seconds_per_unit("large") == pytest.approx(2.0)
    assert tracker.seconds_per_unit("tiny") == 1.0


async def test_a_call_that_loads_its_model_reports_progress_but_learns_nothing() -> None:
    """Climb between the checkpoints as usual, and leave the speed alone."""
    clock = MockMonotonicClock()
    tracker = InferenceProgress(monotonic=clock, seconds_per_unit=1.0, interval_seconds=INTERVAL)
    reports: list[tuple[JobPhase, int]] = []
    await _track(tracker, reports, clock, seconds=30, learn=False)
    assert tracker.seconds_per_unit(MODEL) == 1.0
    assert all(30 < percentage < 85 for _, percentage in reports)
    assert reports


async def test_failed_inference_does_not_change_the_speed() -> None:
    """Learn only from work that finished."""
    clock = MockMonotonicClock()
    tracker = InferenceProgress(monotonic=clock, seconds_per_unit=1.0, interval_seconds=INTERVAL)

    async def record(phase: JobPhase, percentage: int) -> None:
        """Ignore the report."""

    async def fail_inference() -> None:
        """Spend time inside the tracked block, then fail."""
        async with tracker.track(record, phase=JobPhase.INFERENCING, start=30, end=85, units=10, key=MODEL, learn=True):
            clock.advance(seconds=100)
            raise ValueError("inference failed")

    with pytest.raises(ValueError, match="inference failed"):
        await fail_inference()
    assert tracker.seconds_per_unit(MODEL) == 1.0


async def test_failed_write_is_retried_and_a_lost_claim_ends_the_reports() -> None:
    """Survive a database error, and stop reporting once the claim is gone."""
    clock = MockMonotonicClock()
    tracker = InferenceProgress(monotonic=clock, seconds_per_unit=1.0, interval_seconds=INTERVAL)
    attempts: list[int] = []

    async def record(phase: JobPhase, percentage: int) -> None:
        """Fail the first write and lose the claim on the second."""
        attempts.append(percentage)
        if len(attempts) == 1:
            raise ConnectionError("database unavailable")
        raise ClaimLostError("claim lost")

    async with tracker.track(record, phase=JobPhase.INFERENCING, start=30, end=85, units=10, key=MODEL, learn=True):
        clock.advance(seconds=5)
        await eventually(lambda: len(attempts) == 2)
        await asyncio.sleep(INTERVAL * 5)
    assert len(attempts) == 2


async def test_running_job_progress_moves_while_the_model_runs(tmp_path: Path) -> None:
    """Move a real job past its 30% checkpoint during a blocked call, and log the phase once."""
    stt = MockBlockingSttBackend()
    harness = build_harness(tmp_path, stt=stt, progress_interval_seconds=INTERVAL)
    worker = asyncio.create_task(harness.worker.run())
    try:
        command = TranscribeCommand(input_path=write_tone(harness.inputs / "a.wav"))
        job_id = (await harness.jobs.submit(command, owner=OWNER)).job_id
        await eventually(stt.started.is_set)
        harness.monotonic.advance(seconds=1)
        await eventually(lambda: 30 < harness.uow.db.jobs[job_id].progress < 85)
        harness.monotonic.advance(seconds=1)
        first = harness.uow.db.jobs[job_id].progress
        await eventually(lambda: harness.uow.db.jobs[job_id].progress > first)
        stt.release.set()
        await eventually(lambda: harness.uow.db.jobs[job_id].status is JobStatus.COMPLETED)
        phases = [event.phase for event in harness.uow.db.events if event.job_id == job_id and event.phase]
        assert phases.count(JobPhase.INFERENCING) == 1
    finally:
        stt.release.set()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        harness.executor.shutdown()


async def test_transcription_learns_only_once_its_model_is_resident(tmp_path: Path) -> None:
    """Skip the job that loaded the model, then learn from the next one on the same model."""
    harness = build_harness(tmp_path)
    try:
        audio = write_tone(harness.inputs / "a.wav")
        first = await harness.jobs.submit(TranscribeCommand(input_path=audio, model=MODEL), owner=OWNER)
        await harness.run_queued()
        assert harness.uow.db.jobs[first.job_id].status is JobStatus.COMPLETED
        starting_speed = harness.stt_progress.seconds_per_unit("unseen")
        assert harness.stt_progress.seconds_per_unit(MODEL) == starting_speed
        second = await harness.jobs.submit(TranscribeCommand(input_path=audio, model=MODEL), owner=OWNER)
        await harness.run_queued()
        assert harness.uow.db.jobs[second.job_id].status is JobStatus.COMPLETED
        assert harness.stt_progress.seconds_per_unit(MODEL) != starting_speed
    finally:
        harness.executor.shutdown()
