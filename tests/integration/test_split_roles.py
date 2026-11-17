"""The api and worker roles in one test process against PostgreSQL: submit, run, wait, recover and drain."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest

from app.adapters.queue.asyncio.queue import AsyncioJobQueue
from app.adapters.queue.postgres_notify.queue import (
    PostgresAnnouncingJobQueue,
    PostgresListeningJobQueue,
    connector,
)
from app.core.clock import SystemMonotonicClock
from app.db.engine import plain_database_url
from app.db.unit_of_work import SqlUnitOfWorkFactory
from app.domain.errors import CapabilityUnavailableError
from app.domain.jobs import JobError, JobStatus, TranscribeCommand
from app.domain.workers import WorkerStatus
from app.services.runtime_catalog import RegistryRuntimeCatalog
from app.workers.gpu_worker import GpuWorker
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness
from tests.mocks.queue import MockWorkerLock
from tests.mocks.storage import MockAssetStore
from tests.waiting import eventually

pytestmark = pytest.mark.integration
TIMEOUT = 10.0


@dataclass
class Split:
    """An api harness and a worker harness sharing one database and asset store."""

    api: Harness
    worker: Harness
    registry: RegistryRuntimeCatalog
    gpu: GpuWorker
    announcer: PostgresAnnouncingJobQueue


@pytest.fixture
async def split(sql_uow: SqlUnitOfWorkFactory, migrated_database: str, tmp_path: Path) -> AsyncIterator[Split]:
    """Build both roles: the api with no models and the registry catalog, the worker with mock engines."""
    connect = connector(plain_database_url(migrated_database), timeout_seconds=5)
    store = MockAssetStore()
    announcer = PostgresAnnouncingJobQueue(connect=connect, maxsize=100)
    api = build_harness(
        tmp_path / "api", tts=None, installed=set(), registry=True, uow=sql_uow, queue=announcer, store=store
    )
    listener = PostgresListeningJobQueue(connect=connect, local=AsyncioJobQueue(maxsize=100))
    worker = build_harness(tmp_path / "worker", uow=sql_uow, queue=listener, store=store)
    gpu = GpuWorker(
        worker_id="gpu-host:7",
        version="test",
        lock=MockWorkerLock(),
        worker=worker.worker,
        recovery=worker.recovery,
        model_host=worker.model_host,
        runtime=worker.local_runtime,
        record_preload_errors=worker.local_runtime.record_preload_errors,
        uow=sql_uow,
        clock=worker.clock,
        monotonic=SystemMonotonicClock(),
        heartbeat_seconds=0.05,
        poll_interval_seconds=60,
        drain_timeout_seconds=30,
        startup_recovery=True,
        loops=(listener.run,),
    )
    assert isinstance(api.runtime, RegistryRuntimeCatalog)
    announcing = asyncio.create_task(announcer.run())
    try:
        yield Split(api=api, worker=worker, registry=api.runtime, gpu=gpu, announcer=announcer)
    finally:
        announcing.cancel()
        await asyncio.gather(announcing, return_exceptions=True)
        api.executor.shutdown()
        worker.executor.shutdown()


async def test_job_submitted_on_the_api_runs_on_the_worker(split: Split) -> None:
    """Validate from the snapshot, announce, run on the worker, return the result inline, then drain cleanly."""
    with pytest.raises(CapabilityUnavailableError, match="register-only"):
        await split.api.jobs.submit(
            TranscribeCommand(input_path=write_tone(split.api.inputs / "early.wav")), owner=OWNER
        )
    running = asyncio.create_task(split.gpu.run())
    await eventually(lambda: split.gpu.status is WorkerStatus.READY, within=TIMEOUT)
    await split.registry.refresh()
    submission = await split.api.jobs.submit(
        TranscribeCommand(input_path=write_tone(split.api.inputs / "a.wav")), owner=OWNER
    )
    view = await split.api.jobs.wait_for(submission.job_id, owner=OWNER, timeout_seconds=TIMEOUT)
    assert view.status is JobStatus.COMPLETED
    assert set(view.files) == {"txt", "json", "srt", "vtt"}
    assert view.result["text"] == "hello world"
    async with split.api.uow() as uow:
        assert await uow.workers.request_drain("gpu-host:7", now=split.api.clock.now())
    await asyncio.wait_for(running, TIMEOUT)
    async with split.api.uow() as uow:
        row = await uow.workers.get("gpu-host:7")
    assert row is not None
    assert row.status is WorkerStatus.STOPPED
    assert row.drained_at is not None


async def test_api_recovery_runs_while_the_worker_is_stopped(split: Split) -> None:
    """Requeue due retries and resolve stale leases from the api role with no worker running."""
    api = split.api
    await split.gpu.register_only()
    await split.registry.refresh()
    stale = await api.jobs.submit(TranscribeCommand(input_path=write_tone(api.inputs / "s.wav")), owner=OWNER)
    retried = await api.jobs.submit(TranscribeCommand(input_path=write_tone(api.inputs / "r.wav")), owner=OWNER)
    now = api.clock.now()
    async with api.uow() as uow:
        await uow.jobs.claim(stale.job_id, lease_owner="crashed:1", now=now - timedelta(seconds=600))
        claimed = await uow.jobs.claim(retried.job_id, lease_owner="crashed:1", now=now)
        assert claimed is not None
        await uow.jobs.schedule_retry(
            retried.job_id,
            JobError("inference_failed", "boom"),
            claim_token=claimed.claim_token,
            next_attempt_at=now - timedelta(seconds=1),
        )
    await api.recovery.recover()
    for job_id in (stale.job_id, retried.job_id):
        assert (await api.queries.get(job_id, owner=OWNER)).status is JobStatus.QUEUED
