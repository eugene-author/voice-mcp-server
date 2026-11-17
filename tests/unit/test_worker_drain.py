"""The worker role's registry heartbeat and drain protocol: finish, cancel, deadline release, signals and stop."""

import asyncio
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from app.core.clock import SystemMonotonicClock
from app.core.executor import InferenceRunner
from app.domain.errors import ConfigurationError
from app.domain.ids import JobId
from app.domain.jobs import JobStatus, JobType, TranscribeCommand
from app.domain.models import ModelKind
from app.domain.workers import ModelRef, WorkerRecord, WorkerStatus
from app.main import InferenceComponents, _release_inference
from app.workers import gpu_worker
from app.workers.gpu_worker import GpuWorker
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness, host_profile
from tests.mocks.handlers import MockBlockingHandler
from tests.mocks.queue import MockWorkerLock
from tests.mocks.repositories import MockWorkerRepository
from tests.mocks.stt import MockBlockingSttBackend
from tests.waiting import eventually

ROOT = Path(__file__).resolve().parents[2]
WORKER_ID = "gpu-host:42"
TIMEOUT = 5.0


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Return a harness whose transcription handler blocks until the test releases it."""
    harness = build_harness(tmp_path)
    harness.handlers[JobType.SPEECH_TO_TEXT] = MockBlockingHandler(harness.handlers[JobType.SPEECH_TO_TEXT])
    yield harness
    harness.executor.shutdown()


def _blocking(harness: Harness) -> MockBlockingHandler:
    """Return the blocking transcription handler."""
    handler = harness.handlers[JobType.SPEECH_TO_TEXT]
    assert isinstance(handler, MockBlockingHandler)
    return handler


def _gpu_worker(
    harness: Harness,
    *,
    lock: MockWorkerLock | None = None,
    drain_timeout: float = 60.0,
    heartbeat_seconds: float = 0.01,
    preload: tuple[ModelRef, ...] = (),
) -> GpuWorker:
    """Return a worker-role process over the harness with fast heartbeats."""
    return GpuWorker(
        worker_id=WORKER_ID,
        version="test",
        lock=lock or MockWorkerLock(),
        worker=harness.worker,
        recovery=harness.recovery,
        model_host=harness.model_host,
        runtime=harness.local_runtime,
        record_preload_errors=harness.local_runtime.record_preload_errors,
        uow=harness.uow,
        clock=harness.clock,
        monotonic=SystemMonotonicClock(),
        heartbeat_seconds=heartbeat_seconds,
        poll_interval_seconds=60,
        drain_timeout_seconds=drain_timeout,
        startup_recovery=True,
        preload=preload,
    )


@asynccontextmanager
async def _running(gpu: GpuWorker) -> AsyncIterator[asyncio.Task[None]]:
    """Run the worker in the background until it is ready; cancel it if the test leaves it running."""
    task = asyncio.create_task(gpu.run())
    await eventually(lambda: gpu.status is WorkerStatus.READY)
    try:
        yield task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _submit(harness: Harness, name: str = "a.wav") -> JobId:
    """Submit a transcription job."""
    submission = await harness.jobs.submit(TranscribeCommand(input_path=write_tone(harness.inputs / name)), owner=OWNER)
    return submission.job_id


async def _status(harness: Harness, job_id: JobId) -> JobStatus:
    """Return a job's status."""
    return (await harness.queries.get(job_id, owner=OWNER)).status


async def _request_drain(harness: Harness) -> None:
    """Set drain_requested_at as the wake controller does."""
    async with harness.uow() as uow:
        assert await uow.workers.request_drain(WORKER_ID, now=harness.clock.now())


async def _cancel_drain(harness: Harness) -> None:
    """Clear drain_requested_at as the wake controller does."""
    async with harness.uow() as uow:
        assert await uow.workers.cancel_drain(WORKER_ID)


async def test_drain_finishes_the_current_job_then_stops(harness: Harness) -> None:
    """Finish the running job, leave newly queued work for the next worker, unload, unlock and record stopped."""
    lock = MockWorkerLock()
    gpu = _gpu_worker(harness, lock=lock)
    async with _running(gpu) as run:
        running = await _submit(harness)
        await asyncio.wait_for(_blocking(harness).started.wait(), TIMEOUT)
        await eventually(lambda: harness.uow.db.workers[WORKER_ID].status is WorkerStatus.BUSY)
        await _request_drain(harness)
        await eventually(lambda: harness.uow.db.workers[WORKER_ID].status is WorkerStatus.DRAINING)
        waiting = await _submit(harness, "b.wav")
        _blocking(harness).finish.set()
        await asyncio.wait_for(run, TIMEOUT)
    assert await _status(harness, running) is JobStatus.COMPLETED
    assert await _status(harness, waiting) is JobStatus.QUEUED
    row = harness.uow.db.workers[WORKER_ID]
    assert row.status is WorkerStatus.STOPPED
    assert row.drained_at is not None
    assert row.last_activity_at is not None
    assert not lock.held
    assert harness.model_host.loaded_models == {"stt": None, "tts": None}


async def test_cancelled_drain_resumes_claiming(harness: Harness) -> None:
    """Return to busy and keep taking jobs when the controller clears its drain request in time."""
    gpu = _gpu_worker(harness)
    async with _running(gpu) as run:
        first = await _submit(harness)
        await asyncio.wait_for(_blocking(harness).started.wait(), TIMEOUT)
        await _request_drain(harness)
        await eventually(lambda: gpu.status is WorkerStatus.DRAINING)
        await _cancel_drain(harness)
        await eventually(lambda: gpu.status is WorkerStatus.BUSY)
        _blocking(harness).finish.set()
        await eventually(lambda: harness.uow.db.jobs[first].status is JobStatus.COMPLETED)
        second = await _submit(harness, "b.wav")
        await eventually(lambda: harness.uow.db.jobs[second].status is JobStatus.COMPLETED)
        await eventually(lambda: gpu.status is WorkerStatus.READY)
        gpu.request_stop()
        await asyncio.wait_for(run, TIMEOUT)
    assert harness.uow.db.workers[WORKER_ID].status is WorkerStatus.STOPPED


async def test_drain_deadline_releases_the_job_without_using_an_attempt(harness: Harness) -> None:
    """Cancel the handler at the deadline and requeue its job with the attempt given back."""
    gpu = _gpu_worker(harness, drain_timeout=0.05)
    async with _running(gpu) as run:
        job_id = await _submit(harness)
        await asyncio.wait_for(_blocking(harness).started.wait(), TIMEOUT)
        assert harness.uow.db.jobs[job_id].attempt_count == 1
        await _request_drain(harness)
        await asyncio.wait_for(run, TIMEOUT)
    record = harness.uow.db.jobs[job_id]
    assert _blocking(harness).cancelled
    assert (record.status, record.attempt_count, record.lease_owner) == (JobStatus.QUEUED, 0, None)
    assert [event.event_type for event in harness.uow.db.events if event.job_id == job_id][-1] == "released"
    assert harness.uow.db.workers[WORKER_ID].status is WorkerStatus.STOPPED


async def test_signal_drain_cannot_be_cancelled(harness: Harness) -> None:
    """Keep draining after a SIGTERM even when the controller's drain request is cleared."""
    gpu = _gpu_worker(harness)
    async with _running(gpu) as run:
        job_id = await _submit(harness)
        await asyncio.wait_for(_blocking(harness).started.wait(), TIMEOUT)
        await _request_drain(harness)
        await eventually(lambda: gpu.status is WorkerStatus.DRAINING)
        gpu.request_stop()
        await asyncio.sleep(0.05)
        await _cancel_drain(harness)
        await asyncio.sleep(0.05)
        assert gpu.status is WorkerStatus.DRAINING
        _blocking(harness).finish.set()
        await asyncio.wait_for(run, TIMEOUT)
    assert harness.uow.db.jobs[job_id].status is JobStatus.COMPLETED
    assert harness.uow.db.workers[WORKER_ID].status is WorkerStatus.STOPPED


async def test_idle_worker_stops_promptly_on_signal(harness: Harness) -> None:
    """Stop at once on SIGTERM when no job runs."""
    gpu = _gpu_worker(harness)
    async with _running(gpu) as run:
        gpu.request_stop()
        await asyncio.wait_for(run, TIMEOUT)
    assert harness.uow.db.workers[WORKER_ID].status is WorkerStatus.STOPPED


async def test_second_worker_refuses_to_start(harness: Harness) -> None:
    """Refuse to run without the worker lock and register nothing."""
    with pytest.raises(ConfigurationError, match="worker lock"):
        await _gpu_worker(harness, lock=MockWorkerLock(available=False)).run()
    assert harness.uow.db.workers == {}


async def test_preload_loads_models_and_reports_failures(harness: Harness) -> None:
    """Load the requested models before becoming ready; a failing entry is reported, not fatal."""
    preload = (ModelRef(ModelKind.STT, "small"), ModelRef(ModelKind.STT, "large-v3"))
    gpu = _gpu_worker(harness, preload=preload)
    async with _running(gpu):
        snapshot = harness.uow.db.workers[WORKER_ID].capabilities
        assert harness.model_host.loaded_models["stt"] == "small"
        assert snapshot.loaded["stt"] == "small"
        assert len(snapshot.preload_errors) == 1
        assert snapshot.preload_errors[0].startswith("stt:large-v3: ")
        gpu.request_stop()


async def test_register_only_publishes_capabilities_without_loading(harness: Harness) -> None:
    """Record a stopped row carrying the snapshot, with no model loaded and no lock taken."""
    lock = MockWorkerLock()
    snapshot = await _gpu_worker(harness, lock=lock).register_only()
    row = harness.uow.db.workers[WORKER_ID]
    assert (row.status, row.capabilities) == (WorkerStatus.STOPPED, snapshot)
    assert harness.stt.load_count == 0
    assert not lock.held


async def test_abandoned_native_call_does_not_hold_up_stopping(tmp_path: Path) -> None:
    """Release the job, unlock and record stopped while the native call is still blocked on the inference thread."""
    stt = MockBlockingSttBackend()
    harness = build_harness(tmp_path, stt=stt)
    lock = MockWorkerLock()
    gpu = _gpu_worker(harness, lock=lock, drain_timeout=0.05)
    try:
        async with _running(gpu) as run:
            job_id = await _submit(harness)
            await eventually(stt.started.is_set)
            gpu.request_stop()
            await asyncio.wait_for(run, TIMEOUT)
        record = harness.uow.db.jobs[job_id]
        assert not stt.release.is_set()
        assert gpu.abandoned_inference
        assert (record.status, record.attempt_count, record.lease_owner) == (JobStatus.QUEUED, 0, None)
        assert harness.uow.db.workers[WORKER_ID].status is WorkerStatus.STOPPED
        assert not lock.held
    finally:
        stt.release.set()
        harness.executor.shutdown()


def _record_stop_order(harness: Harness, lock: MockWorkerLock, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the order of unloading, unlocking and recording stopped."""
    order: list[str] = []
    unload_all, release, mark_stopped = harness.model_host.unload_all, lock.release, MockWorkerRepository.mark_stopped

    async def unload() -> None:
        """Record the unload and delegate."""
        order.append("unload")
        await unload_all()

    async def unlock() -> None:
        """Record the first release and delegate."""
        if "unlock" not in order:
            order.append("unlock")
        await release()

    async def stopped(repository: MockWorkerRepository, worker_id: str, *, now: datetime) -> None:
        """Record the final status and delegate."""
        order.append("stopped")
        await mark_stopped(repository, worker_id, now=now)

    monkeypatch.setattr(harness.model_host, "unload_all", unload)
    monkeypatch.setattr(lock, "release", unlock)
    monkeypatch.setattr(MockWorkerRepository, "mark_stopped", stopped)
    return order


async def test_normal_drain_unloads_before_recording_stopped(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the order unload, unlock, stopped when no native call was abandoned."""
    lock = MockWorkerLock()
    order = _record_stop_order(harness, lock, monkeypatch)
    gpu = _gpu_worker(harness, lock=lock, preload=(ModelRef(ModelKind.STT, "tiny"),))
    async with _running(gpu) as run:
        gpu.request_stop()
        await asyncio.wait_for(run, TIMEOUT)
    assert order == ["unload", "unlock", "stopped"]
    assert not gpu.abandoned_inference
    assert harness.model_host.loaded_models == {"stt": None, "tts": None}


async def test_failed_unload_still_records_stopped(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlock and record stopped when unloading raises, without treating inference as abandoned."""

    async def fails() -> None:
        """Fail like a native unload error."""
        raise RuntimeError("unload failed")

    monkeypatch.setattr(harness.model_host, "unload_all", fails)
    lock = MockWorkerLock()
    gpu = _gpu_worker(harness, lock=lock)
    async with _running(gpu) as run:
        gpu.request_stop()
        await asyncio.wait_for(run, TIMEOUT)
    assert harness.uow.db.workers[WORKER_ID].status is WorkerStatus.STOPPED
    assert not lock.held
    assert not gpu.abandoned_inference


async def test_unload_that_times_out_is_treated_as_abandoned(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop after STOP_UNLOAD_TIMEOUT_SECONDS, reporting the stuck unload so the process does not join its thread."""

    async def hangs() -> None:
        """Block indefinitely, like an unload queued behind a stuck native call."""
        await asyncio.Event().wait()

    monkeypatch.setattr(harness.model_host, "unload_all", hangs)
    monkeypatch.setattr(gpu_worker, "STOP_UNLOAD_TIMEOUT_SECONDS", 0.01)
    lock = MockWorkerLock()
    gpu = _gpu_worker(harness, lock=lock)
    async with _running(gpu) as run:
        gpu.request_stop()
        await asyncio.wait_for(run, TIMEOUT)
    assert harness.uow.db.workers[WORKER_ID].status is WorkerStatus.STOPPED
    assert not lock.held
    assert gpu.abandoned_inference


async def test_shutdown_without_waiting_returns_during_a_running_call() -> None:
    """Return from shutdown(wait=False) while the inference thread is still busy."""
    runner = InferenceRunner()
    started, release = threading.Event(), threading.Event()

    def block() -> None:
        """Hold the inference thread."""
        started.set()
        release.wait()

    running = asyncio.ensure_future(runner.run(block))
    try:
        await eventually(started.is_set)
        await asyncio.wait_for(asyncio.to_thread(runner.shutdown, wait=False), TIMEOUT)
        assert not running.done()
    finally:
        release.set()
        await running


async def test_abandoned_inference_skips_the_unload_and_the_join(tmp_path: Path) -> None:
    """Skip unload_all and the thread join in _release_inference after a call was abandoned."""
    stt = MockBlockingSttBackend()
    harness = build_harness(tmp_path, stt=stt)
    running = asyncio.ensure_future(harness.executor.run(stt.transcribe, None))
    components = InferenceComponents(
        host=host_profile(),
        stt=harness.stt,
        tts=None,
        denoiser=None,
        model_host=harness.model_host,
        runtime=harness.local_runtime,
        executor=harness.executor,
        denoise_device=None,
        profile="linux-cpu",
    )
    try:
        await eventually(stt.started.is_set)
        await asyncio.wait_for(_release_inference(components, abandoned=True), TIMEOUT)
        assert not running.done()
    finally:
        stt.release.set()
        await asyncio.gather(running, return_exceptions=True)
        harness.executor.shutdown()


def test_run_worker_exits_without_joining_an_abandoned_call() -> None:
    """End the process with code 0, logs flushed, while a native call still holds the inference thread."""
    script = textwrap.dedent("""
        import asyncio, logging, threading
        import app.main
        from app.core.executor import InferenceRunner
        from app.core.settings import load_settings

        async def abandoned(settings, *, preload, register_only):
            runner, started = InferenceRunner(), threading.Event()
            def block():
                started.set()
                threading.Event().wait()
            pending = asyncio.ensure_future(runner.run(block))
            while not started.is_set():
                await asyncio.sleep(0.01)
            pending.cancel()
            runner.shutdown(wait=False)
            logging.getLogger("app.test").warning("abandoned the native call")
            return True

        app.main.run_gpu_worker = abandoned
        app.main.run_worker(load_settings(server={"transport": "stdio"}))
        print("the interpreter kept running")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "the interpreter kept running" not in result.stdout
    assert "abandoned the native call" in result.stdout + result.stderr


def _count_beats(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every worker registry heartbeat."""
    beats: list[float] = []
    beat = MockWorkerRepository.beat

    async def counted(repository: MockWorkerRepository, worker_id: str, **kwargs: Any) -> WorkerRecord | None:
        """Record the beat and delegate."""
        beats.append(time.monotonic())
        return await beat(repository, worker_id, **kwargs)

    monkeypatch.setattr(MockWorkerRepository, "beat", counted)
    return beats


async def test_signal_drain_keeps_the_heartbeat_interval(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Beat at the configured pace while a job drains, not as fast as the loop can spin on the set signal."""
    beats = _count_beats(monkeypatch)
    gpu = _gpu_worker(harness, heartbeat_seconds=0.05)
    async with _running(gpu) as run:
        await _submit(harness)
        await asyncio.wait_for(_blocking(harness).started.wait(), TIMEOUT)
        gpu.request_stop()
        await eventually(lambda: gpu.status is WorkerStatus.DRAINING)
        beats.clear()
        await asyncio.sleep(0.3)
        counted = len(beats)
        _blocking(harness).finish.set()
        await asyncio.wait_for(run, TIMEOUT)
    assert 1 <= counted <= 20


async def test_job_end_wakes_a_signal_drain_before_the_next_heartbeat(harness: Harness) -> None:
    """Stop as soon as the job finishes, however long the heartbeat interval is."""
    gpu = _gpu_worker(harness, heartbeat_seconds=60)
    async with _running(gpu) as run:
        job_id = await _submit(harness)
        await asyncio.wait_for(_blocking(harness).started.wait(), TIMEOUT)
        gpu.request_stop()
        await eventually(lambda: gpu.status is WorkerStatus.DRAINING)
        _blocking(harness).finish.set()
        await asyncio.wait_for(run, TIMEOUT)
    assert harness.uow.db.jobs[job_id].status is JobStatus.COMPLETED


async def test_deadline_wakes_a_signal_drain_before_the_next_heartbeat(harness: Harness) -> None:
    """Release the job at the deadline, however long the heartbeat interval is."""
    gpu = _gpu_worker(harness, heartbeat_seconds=60, drain_timeout=0.05)
    async with _running(gpu) as run:
        job_id = await _submit(harness)
        await asyncio.wait_for(_blocking(harness).started.wait(), TIMEOUT)
        gpu.request_stop()
        await asyncio.wait_for(run, TIMEOUT)
    assert harness.uow.db.jobs[job_id].status is JobStatus.QUEUED


async def test_signal_interrupts_a_controller_drain_before_the_next_heartbeat(harness: Harness) -> None:
    """Turn a controller drain into a signal drain at once; test_signal_drain_cannot_be_cancelled covers the rest."""
    gpu = _gpu_worker(harness)
    async with _running(gpu) as run:
        await _submit(harness)
        await asyncio.wait_for(_blocking(harness).started.wait(), TIMEOUT)
        await _request_drain(harness)
        await eventually(lambda: gpu.status is WorkerStatus.DRAINING)
        gpu._heartbeat = 60
        await asyncio.sleep(0.03)
        gpu.request_stop()
        await eventually(lambda: gpu._drain_by_signal, within=TIMEOUT)
        _blocking(harness).finish.set()
        await asyncio.wait_for(run, TIMEOUT)
    assert harness.uow.db.workers[WORKER_ID].status is WorkerStatus.STOPPED
