"""Ingest across machines on S3 (moto): an api host and a worker host share no filesystem, only the bucket."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from app.adapters.storage.s3.store import S3AssetStore
from app.core.settings import S3Settings
from app.domain.assets import AssetRole
from app.domain.jobs import JobStatus, RegisterVoiceCommand, TranscribeCommand
from app.services.runtime_catalog import RegistryRuntimeCatalog
from app.workers.reconciliation import AssetReconciler
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness, publish_worker
from tests.mocks.clock import MockClock
from tests.mocks.media import MockMedia
from tests.mocks.unit_of_work import MockUnitOfWorkFactory

pytestmark = pytest.mark.s3
BUCKET = "voice-split"
INGEST = {"storage": {"input_transfer": "ingest", "backend": "s3", "s3": {"bucket": BUCKET, "region": "eu-west-1"}}}


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[S3AssetStore]:
    """Return a store on a moto bucket with dummy credentials."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        boto3.client("s3", region_name="eu-west-1").create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
        )
        yield S3AssetStore.from_settings(S3Settings(bucket=BUCKET, region="eu-west-1", prefix="voice-mcp"))


@pytest.fixture
def hosts(tmp_path: Path, store: S3AssetStore) -> Iterator[tuple[Harness, Harness]]:
    """Return an api host (no models, registry catalog) and a worker host sharing only the database and bucket."""
    uow = MockUnitOfWorkFactory()
    api = build_harness(
        tmp_path / "api-host", tts=None, installed=set(), registry=True, uow=uow, store=store, settings_overrides=INGEST
    )
    worker = build_harness(
        tmp_path / "worker-host", uow=uow, store=store, media=MockMedia(duration_seconds=8), settings_overrides=INGEST
    )
    yield api, worker
    api.executor.shutdown()
    worker.executor.shutdown()


async def _run_on_worker(api: Harness, worker: Harness) -> None:
    """Deliver the api's wake-ups to the worker and run them."""
    for job_id in api.queue.notified:
        worker.queue.notify(job_id)
    api.queue.notified.clear()
    await worker.run_queued()


async def test_speech_to_text_and_register_voice_complete_without_a_shared_filesystem(
    hosts: tuple[Harness, Harness],
) -> None:
    """Upload on the api, delete the api's files, run on the worker from S3, and record each original once."""
    api, worker = hosts
    await publish_worker(api.uow, worker.local_runtime.snapshot(), at=api.clock.now())
    assert isinstance(api.runtime, RegistryRuntimeCatalog)
    await api.runtime.refresh()
    talk = write_tone(api.inputs / "talk.wav", seconds=8)
    reference = write_tone(api.inputs / "me.wav", seconds=8)
    stt = await api.jobs.submit(TranscribeCommand(input_path=talk), owner=OWNER)
    voice = await api.jobs.submit(
        RegisterVoiceCommand(
            name="me", reference_audio_path=reference, reference_text="hello world", consent_confirmed=True
        ),
        owner=OWNER,
    )
    talk.unlink()
    reference.unlink()
    await _run_on_worker(api, worker)
    for submission in (stt, voice):
        view = await api.queries.get(submission.job_id, owner=OWNER)
        assert view.status is JobStatus.COMPLETED, view.error
    stt_view = await api.queries.get(stt.job_id, owner=OWNER)
    assert stt_view.files["txt"].startswith(f"s3://{BUCKET}/")
    originals = [record for record in api.uow.db.assets if record.asset.role is AssetRole.INPUT]
    assert sorted(str(record.job_id) for record in originals) == sorted([str(stt.job_id), str(voice.job_id)])
    assert all(record.asset.stored.location.uri.startswith(f"s3://{BUCKET}/") for record in originals)


async def test_queued_job_survives_reconciliation_past_the_retention_window(
    hosts: tuple[Harness, Harness], store: S3AssetStore
) -> None:
    """Reconcile on the api a day after submission, then run the job on the worker from its input in S3."""
    api, worker = hosts
    await publish_worker(api.uow, worker.local_runtime.snapshot(), at=api.clock.now())
    assert isinstance(api.runtime, RegistryRuntimeCatalog)
    await api.runtime.refresh()
    talk = write_tone(api.inputs / "talk.wav", seconds=8)
    submission = await api.jobs.submit(TranscribeCommand(input_path=talk), owner=OWNER)
    talk.unlink()
    a_day_later = MockClock(datetime.now(tz=UTC) + timedelta(hours=25))
    reconciler = AssetReconciler(uow=api.uow, store=store, clock=a_day_later, retention_hours=24, interval_seconds=3600)
    assert await reconciler.reconcile() == 0
    await _run_on_worker(api, worker)
    view = await api.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.COMPLETED, view.error
