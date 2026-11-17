"""storage.input_transfer=ingest: upload at submission, fetch on the worker, record the original once."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.domain.assets import INPUT_LOCATION_FIELD, AssetKey, AssetRole, UploadedInput
from app.domain.errors import InvalidRequestError, PersistenceUnavailableError
from app.domain.jobs import BatchTranscribeCommand, JobStatus, RegisterVoiceCommand, TranscribeCommand
from app.workers.reconciliation import AssetReconciler
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness
from tests.mocks.media import MockMedia

INGEST = {"storage": {"input_transfer": "ingest"}}


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Return a harness in ingest mode."""
    harness = build_harness(tmp_path, settings_overrides=INGEST)
    yield harness
    harness.executor.shutdown()


def _input_uris(harness: Harness) -> list[str]:
    """Return stored objects under any job's inputs/ prefix."""
    return [uri for uri in harness.store.blobs if "/inputs/" in uri]


def _count_puts(harness: Harness, monkeypatch: pytest.MonkeyPatch, *, on_put: Any = None) -> list[str]:
    """Record the key of every upload, calling on_put first."""
    keys: list[str] = []
    original = harness.store.put

    async def put(source: Path, *, key: AssetKey, content_type: str) -> Any:
        """Record the key and delegate."""
        if on_put is not None:
            on_put()
        keys.append(key.value)
        return await original(source, key=key, content_type=content_type)

    monkeypatch.setattr(harness.store, "put", put)
    return keys


async def test_inputs_are_uploaded_before_the_jobs_are_persisted(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upload while no job row exists yet, then persist the job with its input_location."""
    rows_at_upload: list[int] = []
    _count_puts(harness, monkeypatch, on_put=lambda: rows_at_upload.append(len(harness.uow.db.jobs)))
    path = write_tone(harness.inputs / "talk.wav")
    submission = await harness.jobs.submit(TranscribeCommand(input_path=path), owner=OWNER)
    uploaded = UploadedInput.from_payload(harness.uow.db.jobs[submission.job_id].job.input_payload)
    assert rows_at_upload == [0]
    assert uploaded is not None
    assert uploaded.key.value.endswith(f"/jobs/{submission.job_id}/inputs/original.wav")
    assert harness.store.blobs[uploaded.stored.location.uri] == path.read_bytes()


async def test_worker_fetches_the_input_and_records_it_once(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run from the fetched copy when the submitted path is gone, and record the uploaded object as the original."""
    keys = _count_puts(harness, monkeypatch)
    path = write_tone(harness.inputs / "talk.wav")
    submission = await harness.jobs.submit(TranscribeCommand(input_path=path), owner=OWNER)
    path.unlink()
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.COMPLETED
    assert sum("/inputs/" in key for key in keys) == 1
    originals = [a for a in harness.uow.db.assets if a.asset.role is AssetRole.INPUT]
    assert len(originals) == 1
    uploaded = UploadedInput.from_payload(harness.uow.db.jobs[submission.job_id].job.input_payload)
    assert uploaded is not None
    assert originals[0].asset.stored == uploaded.stored


async def test_registration_reads_the_uploaded_reference(tmp_path: Path) -> None:
    """Register a voice from the uploaded recording after the submitted file is gone."""
    harness = build_harness(tmp_path, settings_overrides=INGEST, media=MockMedia(duration_seconds=8))
    try:
        await _register(harness)
    finally:
        harness.executor.shutdown()


async def _register(harness: Harness) -> None:
    """Submit a registration, remove the local file and run it."""
    reference = write_tone(harness.inputs / "me.wav", seconds=8)
    submission = await harness.jobs.submit(
        RegisterVoiceCommand(
            name="me", reference_audio_path=reference, reference_text="hello world", consent_confirmed=True
        ),
        owner=OWNER,
    )
    reference.unlink()
    await harness.run_queued()
    assert (await harness.queries.get(submission.job_id, owner=OWNER)).status is JobStatus.COMPLETED
    roles = sorted(str(a.asset.role) for a in harness.uow.db.assets if a.job_id == submission.job_id)
    assert roles == ["input", "voice_reference"]
    assert len(_input_uris(harness)) == 1


async def test_batches_upload_every_file_within_the_limit(harness: Harness) -> None:
    """Upload each child of a batch; the parent carries no input."""
    for name in ("a.wav", "b.wav", "c.wav"):
        write_tone(harness.inputs / name)
    submission = await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
    children = [r for r in harness.uow.db.jobs.values() if r.job.parent_job_id == submission.job_id]
    assert len(children) == 3
    assert all(INPUT_LOCATION_FIELD in child.job.input_payload for child in children)
    assert INPUT_LOCATION_FIELD not in harness.uow.db.jobs[submission.job_id].job.input_payload
    assert len(_input_uris(harness)) == 3


async def test_batch_over_the_upload_limit_fails_before_uploading(tmp_path: Path) -> None:
    """Reject a batch whose inputs exceed audio.max_batch_upload_bytes with invalid_request, uploading nothing."""
    harness = build_harness(tmp_path, settings_overrides={**INGEST, "audio": {"max_batch_upload_bytes": 100_000}})
    try:
        for name in ("a.wav", "b.wav"):
            write_tone(harness.inputs / name)
        with pytest.raises(InvalidRequestError, match="max_batch_upload_bytes"):
            await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
        assert harness.store.blobs == {}
        assert harness.uow.db.jobs == {}
    finally:
        harness.executor.shutdown()


@pytest.mark.parametrize("mode", ["shared_path", "ingest"])
async def test_oversized_batch_file_creates_nothing_in_either_mode(tmp_path: Path, mode: str) -> None:
    """Reject the whole batch, like the single submission, with no job rows and no uploads."""
    overrides = {"storage": {"input_transfer": mode}, "audio": {"max_input_bytes": 1_000}}
    harness = build_harness(tmp_path, settings_overrides=overrides)
    try:
        fine = harness.inputs / "fine.wav"
        fine.write_bytes(b"x" * 1_000)
        big = write_tone(harness.inputs / "big.wav")
        with pytest.raises(InvalidRequestError, match="exceeds the maximum input size"):
            await harness.jobs.submit(TranscribeCommand(input_path=big), owner=OWNER)
        with pytest.raises(InvalidRequestError, match=r"big\.wav exceeds the maximum input size"):
            await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
        assert harness.uow.db.jobs == {}
        assert harness.store.blobs == {}
        assert harness.queue.notified == []
    finally:
        harness.executor.shutdown()


async def test_repeated_request_id_does_not_upload_again(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the earlier job for a known request_id before uploading anything."""
    keys = _count_puts(harness, monkeypatch)
    path = write_tone(harness.inputs / "talk.wav")
    first = await harness.jobs.submit(TranscribeCommand(input_path=path, request_id="r1"), owner=OWNER)
    again = await harness.jobs.submit(TranscribeCommand(input_path=path, request_id="r1"), owner=OWNER)
    assert (again.job_id, again.deduplicated) == (first.job_id, True)
    assert len(keys) == 1


async def test_uploads_orphaned_by_a_failed_commit_are_reconciled(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave no row when persisting fails after upload, and let reconciliation remove the orphan."""

    def database_goes_down() -> None:
        """Fail the commit that follows the upload."""
        harness.uow.available = False

    _count_puts(harness, monkeypatch, on_put=database_goes_down)
    with pytest.raises(PersistenceUnavailableError):
        await harness.jobs.submit(TranscribeCommand(input_path=write_tone(harness.inputs / "a.wav")), owner=OWNER)
    assert len(_input_uris(harness)) == 1
    harness.uow.available = True
    reconciler = AssetReconciler(
        uow=harness.uow, store=harness.store, clock=harness.clock, retention_hours=24, interval_seconds=3600
    )
    assert await reconciler.reconcile() == 1
    assert _input_uris(harness) == []


async def test_shared_path_mode_never_uploads_at_submission(tmp_path: Path) -> None:
    """Keep today's behaviour by default: the worker reads the path and preserves the original at completion."""
    harness = build_harness(tmp_path)
    try:
        submission = await harness.jobs.submit(
            TranscribeCommand(input_path=write_tone(harness.inputs / "a.wav")), owner=OWNER
        )
        assert harness.store.blobs == {}
        assert INPUT_LOCATION_FIELD not in harness.uow.db.jobs[submission.job_id].job.input_payload
        await harness.run_queued()
        assert len(_input_uris(harness)) == 1
    finally:
        harness.executor.shutdown()
