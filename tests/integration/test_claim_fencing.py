"""Claim fencing against real PostgreSQL: conditional updates under real concurrency, and the claim_token migration."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.repositories.jobs import SqlJobRepository
from app.db.unit_of_work import SqlUnitOfWorkFactory
from app.domain.ids import ClaimToken, JobId
from app.domain.jobs import ClaimedJob, JobCompletion, JobError, JobPhase, JobStatus, NewJob
from tests.integration.conftest import alembic_config
from tests.integration.test_sql_repositories import NOW, new_job

pytestmark = pytest.mark.integration
SAME_PROCESS = "gpu-host:1"
LATER = NOW + timedelta(minutes=10)
ERROR = JobError("inference_failed", "stale")
V8_REVISION = "4b8e2c6d1f73"
RACE_ROUNDS = 25

type StaleWrite = Callable[[SqlJobRepository, JobId, ClaimToken], Awaitable[bool]]

STALE_WRITES: dict[str, StaleWrite] = {
    "complete": lambda jobs, job_id, token: jobs.complete(
        job_id, JobCompletion(result={"from": "stale"}), claim_token=token, now=LATER
    ),
    "fail": lambda jobs, job_id, token: jobs.fail(job_id, ERROR, claim_token=token, now=LATER),
    "schedule_retry": lambda jobs, job_id, token: jobs.schedule_retry(
        job_id, ERROR, claim_token=token, next_attempt_at=LATER
    ),
    "set_phase": lambda jobs, job_id, token: jobs.set_phase(
        job_id, claim_token=token, phase=JobPhase.PUBLISHING, progress=92, now=LATER
    ),
    "heartbeat": lambda jobs, job_id, token: jobs.heartbeat(job_id, claim_token=token, now=LATER),
    "release": lambda jobs, job_id, token: jobs.release(job_id, claim_token=token),
}


async def _claim(sql_uow: SqlUnitOfWorkFactory, job: NewJob) -> ClaimedJob:
    """Claim a job that must be claimable, always from the same process."""
    async with sql_uow() as uow:
        claimed = await uow.jobs.claim(job.id, lease_owner=SAME_PROCESS, now=NOW)
    assert claimed is not None
    return claimed


async def _expire_and_requeue(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Run recovery's writes: interrupt every expired lease and requeue the jobs."""
    async with sql_uow() as uow:
        await uow.jobs.interrupt_stale(heartbeat_before=LATER)
        await uow.jobs.resolve_interrupted(now=LATER)


async def _lost_claim(sql_uow: SqlUnitOfWorkFactory) -> tuple[NewJob, ClaimedJob, ClaimedJob]:
    """Return a job, its lost claim and the newer claim from the same process."""
    job = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
    stale = await _claim(sql_uow, job)
    await _expire_and_requeue(sql_uow)
    return job, stale, await _claim(sql_uow, job)


async def _row(engine: AsyncEngine, job_id: JobId) -> dict[str, Any]:
    """Return every worker-writable column of a job row."""
    async with engine.connect() as connection:
        row = await connection.execute(
            text(
                "SELECT status, phase, progress, attempt_count, heartbeat_at, lease_owner, claim_token, "
                "result_payload, error_code, next_attempt_at, finished_at FROM jobs WHERE id = :id"
            ),
            {"id": job_id},
        )
        return dict(row.mappings().one())


@pytest.mark.parametrize("write", STALE_WRITES, ids=str)
async def test_stale_write_changes_no_column(sql_uow: SqlUnitOfWorkFactory, engine: AsyncEngine, write: str) -> None:
    """Reject each worker-owned write of a lost claim and leave the newer claim's row byte for byte."""
    job, stale, _ = await _lost_claim(sql_uow)
    before = await _row(engine, job.id)
    async with sql_uow() as uow:
        assert not await STALE_WRITES[write](uow.jobs, job.id, stale.claim_token)  # type: ignore[arg-type]
    assert await _row(engine, job.id) == before


async def test_reclaim_by_the_same_owner_leaves_the_old_token_invalid(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Fence by token, not by owner: both claims have the same lease owner."""
    job, stale, current = await _lost_claim(sql_uow)
    async with sql_uow() as uow:
        assert not await uow.jobs.heartbeat(job.id, claim_token=stale.claim_token, now=LATER)
        assert await uow.jobs.heartbeat(job.id, claim_token=current.claim_token, now=LATER)


async def test_release_and_reclaim_issue_a_new_token_for_the_same_attempt(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Invalidate a released claim even though the refund makes the attempt count repeat."""
    job = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
    released = await _claim(sql_uow, job)
    async with sql_uow() as uow:
        assert await uow.jobs.release(job.id, claim_token=released.claim_token)
    again = await _claim(sql_uow, job)
    assert again.attempt_count == released.attempt_count
    assert again.claim_token != released.claim_token
    async with sql_uow() as uow:
        assert not await uow.jobs.complete(job.id, JobCompletion(), claim_token=released.claim_token, now=LATER)


async def test_recovery_and_operators_need_no_claim_token(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Keep recovery's and the CLI's authority separate from an executing worker's."""
    job = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
    claimed = await _claim(sql_uow, job)
    async with sql_uow() as uow:
        await uow.jobs.schedule_retry(job.id, ERROR, claim_token=claimed.claim_token, next_attempt_at=LATER)
    async with sql_uow() as uow:
        assert await uow.jobs.set_status(job.id, JobStatus.CANCELLED, now=LATER)
    interrupted = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(interrupted)
    await _claim(sql_uow, interrupted)
    async with sql_uow() as uow:
        assert await uow.jobs.interrupt_stale(heartbeat_before=LATER) == [interrupted.id]


async def test_concurrent_finishes_are_won_by_the_current_claim(
    sql_uow: SqlUnitOfWorkFactory, engine: AsyncEngine
) -> None:
    """Keep the current claim's result whichever concurrent transaction reaches PostgreSQL first."""

    async def finish(job: NewJob, claim: ClaimedJob, source: str) -> bool:
        """Complete the job from its own connection."""
        async with sql_uow() as uow:
            completion = JobCompletion(result={"from": source})
            return await uow.jobs.complete(job.id, completion, claim_token=claim.claim_token, now=LATER)

    for _ in range(RACE_ROUNDS):
        job, stale, current = await _lost_claim(sql_uow)
        results = await asyncio.gather(finish(job, stale, "stale"), finish(job, current, "current"))
        assert results == [False, True]
        assert (await _row(engine, job.id))["result_payload"] == {"from": "current"}


async def test_completion_racing_a_reclaim_never_applies_to_the_new_claim(
    sql_uow: SqlUnitOfWorkFactory, engine: AsyncEngine
) -> None:
    """Let PostgreSQL order a completion against recovery plus reclaim; exactly one of them may take effect."""

    async def reclaim(job: NewJob) -> ClaimedJob | None:
        """Requeue and claim in one transaction, as fast as a worker can."""
        async with sql_uow() as uow:
            await uow.jobs.interrupt_stale(heartbeat_before=LATER)
            await uow.jobs.resolve_interrupted(now=LATER)
            return await uow.jobs.claim(job.id, lease_owner=SAME_PROCESS, now=LATER)

    async def finish(job: NewJob, claim: ClaimedJob) -> bool:
        """Complete from the claim that is being taken away."""
        async with sql_uow() as uow:
            return await uow.jobs.complete(
                job.id, JobCompletion(result={"from": "a"}), claim_token=claim.claim_token, now=LATER
            )

    completions = set()
    for _ in range(RACE_ROUNDS):
        job = new_job()
        async with sql_uow() as uow:
            await uow.jobs.add(job)
        claim = await _claim(sql_uow, job)
        completed, newer = await asyncio.gather(finish(job, claim), reclaim(job))
        row = await _row(engine, job.id)
        assert completed != (newer is not None)
        if completed:
            assert row["status"] == JobStatus.COMPLETED.value
        else:
            assert newer is not None
            assert (row["status"], row["claim_token"]) == (JobStatus.RUNNING.value, newer.claim_token)
            assert row["result_payload"] is None
        completions.add(completed)
    assert completions


async def test_migration_keeps_v8_rows_and_fences_running_ones(
    sql_uow: SqlUnitOfWorkFactory, engine: AsyncEngine
) -> None:
    """Upgrade queued, running, failed and completed V8 rows, then show each behaves under the new code."""
    queued, running, failed, completed = (new_job() for _ in range(4))
    async with sql_uow() as uow:
        for job in (queued, running, failed, completed):
            await uow.jobs.add(job)
    old_running = await _claim(sql_uow, running)
    done, bad = await _claim(sql_uow, completed), await _claim(sql_uow, failed)
    async with sql_uow() as uow:
        await uow.jobs.complete(
            completed.id, JobCompletion(result={"kept": True}), claim_token=done.claim_token, now=NOW
        )
        await uow.jobs.fail(failed.id, ERROR, claim_token=bad.claim_token, now=NOW)
    config = alembic_config()
    try:
        await asyncio.to_thread(command.downgrade, config, V8_REVISION)
        async with engine.connect() as connection:
            columns = await connection.scalars(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'jobs'")
            )
            assert "claim_token" not in set(columns)
    finally:
        await asyncio.to_thread(command.upgrade, config, "head")
    rows = {job.id: await _row(engine, job.id) for job in (queued, running, failed, completed)}
    assert [rows[job.id]["status"] for job in (queued, running, failed, completed)] == [
        "queued",
        "running",
        "failed",
        "completed",
    ]
    assert all(row["claim_token"] is None for row in rows.values())
    assert rows[completed.id]["result_payload"] == {"kept": True}
    async with sql_uow() as uow:
        assert not await uow.jobs.heartbeat(running.id, claim_token=old_running.claim_token, now=LATER)
    await _expire_and_requeue(sql_uow)
    assert (await _claim(sql_uow, queued)).claim_token is not None
    reclaimed = await _claim(sql_uow, running)
    async with sql_uow() as uow:
        assert await uow.jobs.complete(running.id, JobCompletion(), claim_token=reclaimed.claim_token, now=LATER)
