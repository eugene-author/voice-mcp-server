"""Worker registry, job demand and lease-guarded release against real PostgreSQL."""

import asyncio
from datetime import timedelta

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.unit_of_work import SqlUnitOfWorkFactory
from app.domain.ids import new_claim_token
from app.domain.jobs import JobError, JobStatus, JobType
from app.domain.workers import CapabilitySnapshot, JobDemand, WorkerRecord, WorkerStatus
from tests.harness import build_harness
from tests.integration.conftest import alembic_config
from tests.integration.test_sql_repositories import NOW, new_job

pytestmark = pytest.mark.integration
LEASE = timedelta(seconds=120)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> CapabilitySnapshot:
    """Return a capability snapshot of the mock worker runtime."""
    harness = build_harness(tmp_path_factory.mktemp("snapshot"), denoiser=True)
    try:
        return harness.local_runtime.snapshot()
    finally:
        harness.executor.shutdown()


def _worker(
    snapshot: CapabilitySnapshot, worker_id: str, *, started: int, beat: int, status: WorkerStatus
) -> WorkerRecord:
    """Return a worker row started and last seen the given number of seconds after NOW."""
    return WorkerRecord(
        id=worker_id,
        status=status,
        capabilities=snapshot,
        started_at=NOW + timedelta(seconds=started),
        heartbeat_at=NOW + timedelta(seconds=beat),
        version="1.15",
    )


async def test_register_beat_and_drain_round_trip(sql_uow: SqlUnitOfWorkFactory, snapshot: CapabilitySnapshot) -> None:
    """Store the snapshot, update status and activity, and set and clear a drain request once each."""
    async with sql_uow() as uow:
        await uow.workers.register(_worker(snapshot, "gpu:1", started=0, beat=0, status=WorkerStatus.STARTING))
    later = NOW + timedelta(seconds=15)
    async with sql_uow() as uow:
        beaten = await uow.workers.beat("gpu:1", status=WorkerStatus.READY, now=later, last_activity_at=later)
        assert await uow.workers.request_drain("gpu:1", now=later)
        assert not await uow.workers.request_drain("gpu:1", now=later)
        missing = await uow.workers.beat("gone:2", status=WorkerStatus.READY, now=later, last_activity_at=None)
    assert beaten is not None
    assert beaten.capabilities == snapshot
    assert (beaten.status, beaten.heartbeat_at, beaten.last_activity_at) == (WorkerStatus.READY, later, later)
    assert missing is None
    async with sql_uow() as uow:
        requested = await uow.workers.get("gpu:1")
        assert await uow.workers.cancel_drain("gpu:1")
        assert not await uow.workers.cancel_drain("gpu:1")
        await uow.workers.mark_stopped("gpu:1", now=later)
        stopped = await uow.workers.get("gpu:1")
    assert requested is not None
    assert requested.drain_requested_at == later
    assert stopped is not None
    assert (stopped.status, stopped.drained_at, stopped.drain_requested_at) == (WorkerStatus.STOPPED, later, None)


async def test_active_is_the_newest_fresh_worker(sql_uow: SqlUnitOfWorkFactory, snapshot: CapabilitySnapshot) -> None:
    """Pick the most recently started worker with a fresh heartbeat; latest ignores freshness and status."""
    async with sql_uow() as uow:
        await uow.workers.register(_worker(snapshot, "old:1", started=0, beat=300, status=WorkerStatus.READY))
        await uow.workers.register(_worker(snapshot, "new:2", started=100, beat=290, status=WorkerStatus.BUSY))
        await uow.workers.register(_worker(snapshot, "stale:3", started=200, beat=10, status=WorkerStatus.READY))
        await uow.workers.register(_worker(snapshot, "done:4", started=250, beat=310, status=WorkerStatus.STOPPED))
    cutoff = NOW + timedelta(seconds=200)
    async with sql_uow() as uow:
        active = await uow.workers.active(heartbeat_after=cutoff)
        latest = await uow.workers.latest()
        none_fresh = await uow.workers.active(heartbeat_after=NOW + timedelta(seconds=400))
    assert active is not None
    assert active.id == "new:2"
    assert latest is not None
    assert latest.id == "done:4"
    assert none_fresh is None


async def test_delete_stopped_removes_only_old_stopped_rows(
    sql_uow: SqlUnitOfWorkFactory, snapshot: CapabilitySnapshot
) -> None:
    """Delete stopped rows older than the cutoff and keep running or recent ones."""
    async with sql_uow() as uow:
        await uow.workers.register(_worker(snapshot, "old:1", started=0, beat=0, status=WorkerStatus.STOPPED))
        await uow.workers.register(_worker(snapshot, "new:2", started=0, beat=100, status=WorkerStatus.STOPPED))
        await uow.workers.register(_worker(snapshot, "run:3", started=0, beat=0, status=WorkerStatus.READY))
    async with sql_uow() as uow:
        removed = await uow.workers.delete_stopped(before=NOW + timedelta(seconds=50))
        remaining = {worker.id for worker in [await uow.workers.get(i) for i in ("old:1", "new:2", "run:3")] if worker}
    assert removed == 1
    assert remaining == {"new:2", "run:3"}


async def test_delete_stopped_always_keeps_the_newest_row(
    sql_uow: SqlUnitOfWorkFactory, snapshot: CapabilitySnapshot
) -> None:
    """Keep exactly the newest row when every stopped row is past the cutoff."""
    async with sql_uow() as uow:
        for name, beat in (("a:1", 0), ("b:2", 10), ("c:3", 20)):
            await uow.workers.register(_worker(snapshot, name, started=0, beat=beat, status=WorkerStatus.STOPPED))
    async with sql_uow() as uow:
        removed = await uow.workers.delete_stopped(before=NOW + timedelta(days=30))
        latest = await uow.workers.latest()
        remaining = [worker for worker in [await uow.workers.get(i) for i in ("a:1", "b:2", "c:3")] if worker]
    assert removed == 2
    assert latest is not None
    assert [worker.id for worker in remaining] == [latest.id] == ["c:3"]


async def test_delete_stopped_keeps_a_newer_row_and_removes_the_older(
    sql_uow: SqlUnitOfWorkFactory, snapshot: CapabilitySnapshot
) -> None:
    """Delete only the older of two stopped rows past the cutoff."""
    async with sql_uow() as uow:
        await uow.workers.register(_worker(snapshot, "old:1", started=0, beat=0, status=WorkerStatus.STOPPED))
        await uow.workers.register(_worker(snapshot, "new:2", started=0, beat=100, status=WorkerStatus.STOPPED))
    async with sql_uow() as uow:
        removed = await uow.workers.delete_stopped(before=NOW + timedelta(seconds=200))
        assert await uow.workers.get("old:1") is None
        assert await uow.workers.get("new:2") is not None
    assert removed == 1


async def test_demand_counts_runnable_and_fresh_running_jobs(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Count queued jobs, due retries and stale running jobs as runnable; do not count batch parents."""
    queued, due, waiting, fresh, stale, done = (new_job() for _ in range(6))
    parent = new_job(type=JobType.TRANSCRIBE_BATCH)
    async with sql_uow() as uow:
        for job in (queued, due, waiting, fresh, stale, parent, done):
            await uow.jobs.add(job)
    async with sql_uow() as uow:
        tokens = {}
        for job, at in ((due, NOW), (waiting, NOW), (fresh, NOW + LEASE), (stale, NOW), (done, NOW)):
            claimed = await uow.jobs.claim(job.id, lease_owner="gpu:1", now=at)
            assert claimed is not None
            tokens[job.id] = claimed.claim_token
        error = JobError("inference_failed", "x")
        await uow.jobs.schedule_retry(
            due.id, error, claim_token=tokens[due.id], next_attempt_at=NOW + timedelta(seconds=5)
        )
        await uow.jobs.schedule_retry(
            waiting.id, error, claim_token=tokens[waiting.id], next_attempt_at=NOW + timedelta(hours=1)
        )
        await uow.jobs.fail(done.id, error, claim_token=tokens[done.id], now=NOW)
    async with sql_uow() as uow:
        demand = await uow.jobs.demand(now=NOW + LEASE, heartbeat_before=NOW + timedelta(seconds=60))
    assert demand == JobDemand(runnable=3, running=1)


async def test_release_requeues_without_using_the_attempt(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Return a running job to queued only for its claim, giving back the attempt and clearing the lease."""
    job = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
        claimed = await uow.jobs.claim(job.id, lease_owner="gpu:1", now=NOW)
    assert claimed is not None
    async with sql_uow() as uow:
        assert not await uow.jobs.release(job.id, claim_token=new_claim_token())
        assert await uow.jobs.release(job.id, claim_token=claimed.claim_token)
        assert not await uow.jobs.release(job.id, claim_token=claimed.claim_token)
    async with sql_uow() as uow:
        view = await uow.jobs.get(job.id, None)
        again = await uow.jobs.claim(job.id, lease_owner="gpu:3", now=NOW)
    assert view is not None
    assert (view.status, view.progress, view.phase) == (JobStatus.QUEUED, 0, None)
    assert again is not None
    assert again.attempt_count == 1


async def test_workers_migration_downgrades_and_upgrades(engine: AsyncEngine) -> None:
    """Drop and recreate the workers table through its own revision."""
    config = alembic_config()
    try:
        await asyncio.to_thread(command.downgrade, config, "9e3d5b7a1c40")
        async with engine.connect() as connection:
            assert not await connection.scalar(text("SELECT to_regclass('workers') IS NOT NULL"))
    finally:
        await asyncio.to_thread(command.upgrade, config, "head")
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT to_regclass('workers') IS NOT NULL"))
