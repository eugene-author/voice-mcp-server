"""Orphaned asset reconciliation."""

from datetime import datetime, timedelta
from pathlib import Path

from app.domain.assets import UploadedInput
from app.domain.jobs import JobError, TranscribeCommand
from app.domain.workers import WorkerStatus
from app.workers.reconciliation import AssetReconciler
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness, publish_worker

STOPPED = WorkerStatus.STOPPED
RETENTION_SECONDS = 24 * 3600


async def test_removes_only_old_unreferenced_assets(harness: Harness, tmp_path: Path) -> None:
    """Delete stale objects without rows and keep recorded or recent ones."""
    path = write_tone(harness.inputs / "a.wav")
    await harness.jobs.submit(TranscribeCommand(input_path=path), owner=OWNER)
    await harness.run_queued()
    recorded = set(harness.store.blobs)
    harness.store.blobs["/assets/users/x/jobs/y/outputs/orphan.txt"] = b"left behind"
    harness.store.modified_at["/assets/users/x/jobs/y/outputs/orphan.txt"] = datetime.min
    harness.store.blobs["/assets/users/x/jobs/z/outputs/recent.txt"] = b"still uploading"
    harness.store.modified_at["/assets/users/x/jobs/z/outputs/recent.txt"] = datetime.max - timedelta(days=1)
    reconciler = AssetReconciler(
        uow=harness.uow, store=harness.store, clock=harness.clock, retention_hours=24, interval_seconds=3600
    )
    assert await reconciler.reconcile() == 1
    assert recorded <= set(harness.store.blobs)
    assert "/assets/users/x/jobs/z/outputs/recent.txt" in harness.store.blobs
    assert "/assets/users/x/jobs/y/outputs/orphan.txt" not in harness.store.blobs


async def test_removes_week_old_stopped_worker_rows(harness: Harness) -> None:
    """Delete stopped registry rows older than seven days and keep recent or live ones."""
    now = harness.clock.now()
    snapshot = harness.local_runtime.snapshot()
    await publish_worker(harness.uow, snapshot, at=now - timedelta(days=8), worker_id="old:1", status=STOPPED)
    await publish_worker(harness.uow, snapshot, at=now - timedelta(days=1), worker_id="recent:2", status=STOPPED)
    await publish_worker(harness.uow, snapshot, at=now - timedelta(days=8), worker_id="hung:3")
    reconciler = AssetReconciler(
        uow=harness.uow, store=harness.store, clock=harness.clock, retention_hours=24, interval_seconds=3600
    )
    await reconciler.reconcile()
    assert set(harness.uow.db.workers) == {"recent:2", "hung:3"}


async def test_keeps_the_uploaded_input_until_its_job_is_terminal(tmp_path: Path) -> None:
    """Keep a queued job's input past the retention window and delete it once the job failed unrecorded."""
    harness = build_harness(tmp_path, settings_overrides={"storage": {"input_transfer": "ingest"}})
    reconciler = AssetReconciler(
        uow=harness.uow, store=harness.store, clock=harness.clock, retention_hours=24, interval_seconds=3600
    )
    try:
        path = write_tone(harness.inputs / "a.wav")
        submission = await harness.jobs.submit(TranscribeCommand(input_path=path), owner=OWNER)
        uploaded = UploadedInput.from_payload(harness.uow.db.jobs[submission.job_id].job.input_payload)
        assert uploaded is not None
        uri = uploaded.stored.location.uri
        harness.store.modified_at[uri] = harness.clock.now().replace(tzinfo=None)
        harness.clock.advance(seconds=RETENTION_SECONDS + 3600)
        assert await reconciler.reconcile() == 0
        assert uri in harness.store.blobs
        async with harness.uow() as uow:
            now = harness.clock.now()
            claim = await uow.jobs.claim(submission.job_id, lease_owner="worker", now=now)
            assert claim is not None
            await uow.jobs.fail(
                submission.job_id, JobError("media_error", "unreadable"), claim_token=claim.claim_token, now=now
            )
        assert await reconciler.reconcile() == 1
        assert uri not in harness.store.blobs
    finally:
        harness.executor.shutdown()
