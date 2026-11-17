"""Batch transcription: parent/child creation, idempotent aggregation and the manifest."""

import json
import threading
from pathlib import Path

import pytest

from app.domain.errors import InvalidRequestError
from app.domain.jobs import BatchTranscribeCommand, JobFilter, JobStatus, JobType
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness
from tests.mocks.media import MockMedia
from tests.mocks.storage import MockAssetStore


def _files(harness: Harness, names: list[str]) -> None:
    """Write audio files into the allowed input directory."""
    for name in names:
        write_tone(harness.inputs / name)


async def test_batch_creates_children_and_completes_with_manifest(harness: Harness) -> None:
    """Create one child per file, leave the parent unclaimed, and publish a manifest when done."""
    _files(harness, ["a.wav", "b.wav", "c.wav"])
    submission = await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
    assert submission.job_id not in harness.queue.notified
    assert len(harness.queue.notified) == 3
    await harness.run_queued()
    parent = await harness.queries.get(submission.job_id, owner=OWNER)
    assert parent.status is JobStatus.COMPLETED
    assert parent.counts is not None
    assert parent.counts.to_dict() == {"total": 3, "completed": 3, "failed": 0, "queued": 0}
    manifest_uri = parent.files["manifest"]
    manifest = json.loads(harness.store.blobs[manifest_uri])
    assert [Path(item["input_path"]).name for item in manifest["items"]] == ["a.wav", "b.wav", "c.wav"]
    children = await harness.queries.list(
        owner=OWNER, job_filter=JobFilter(parent_job_id=submission.job_id), limit=None, cursor=None
    )
    assert {child.type for child in children.items} == {JobType.SPEECH_TO_TEXT}


async def test_partial_failure_is_completed_with_errors(tmp_path: Path) -> None:
    """Record individual failures and finish as completed_with_errors."""
    harness = build_harness(tmp_path, media=MockMedia(failing_names=frozenset({"bad.wav"})))
    _files(harness, ["bad.wav", "good.wav"])
    submission = await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
    await harness.run_queued()
    parent = await harness.queries.get(submission.job_id, owner=OWNER)
    assert parent.status is JobStatus.COMPLETED_WITH_ERRORS
    manifest = json.loads(harness.store.blobs[parent.files["manifest"]])
    errors = {Path(item["input_path"]).name: item["error"] for item in manifest["items"]}
    assert errors["bad.wav"]["code"] == "media_error"
    assert errors["good.wav"] is None
    harness.executor.shutdown()


async def test_all_children_failing_fails_the_parent(tmp_path: Path) -> None:
    """Fail the parent when no child succeeded."""
    harness = build_harness(tmp_path, media=MockMedia(fail_probe=True))
    _files(harness, ["a.wav", "b.wav"])
    submission = await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
    await harness.run_queued()
    assert (await harness.queries.get(submission.job_id, owner=OWNER)).status is JobStatus.FAILED
    harness.executor.shutdown()


async def test_child_retry_counts_once(tmp_path: Path) -> None:
    """Keep counts exact when a child goes through retry_wait before succeeding."""
    harness = build_harness(tmp_path, store=MockAssetStore(fail_puts=1))
    _files(harness, ["a.wav", "b.wav"])
    submission = await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
    await harness.run_queued()
    midway = await harness.queries.get(submission.job_id, owner=OWNER)
    assert midway.status is JobStatus.RUNNING
    assert midway.counts is not None
    assert midway.counts.to_dict() == {"total": 2, "completed": 1, "failed": 0, "queued": 1}
    harness.clock.advance(seconds=10)
    await harness.recovery.recover()
    await harness.run_queued()
    final = await harness.queries.get(submission.job_id, owner=OWNER)
    assert final.status is JobStatus.COMPLETED
    assert final.counts is not None
    assert final.counts.to_dict() == {"total": 2, "completed": 2, "failed": 0, "queued": 0}
    harness.executor.shutdown()


async def test_recovery_finalizes_parent_after_crash(harness: Harness) -> None:
    """Finalize a parent whose children all finished before aggregation ran."""
    _files(harness, ["a.wav"])
    submission = await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
    child_id = harness.queue.notified.pop()
    job = await harness.transitions.claim(child_id, lease_owner="w")
    assert job is not None
    completion = await harness.worker._execute(job)
    await harness.worker._finalize(job, completion)
    assert (await harness.queries.get(submission.job_id, owner=OWNER)).status is JobStatus.QUEUED
    await harness.recovery.recover()
    assert (await harness.queries.get(submission.job_id, owner=OWNER)).status is JobStatus.COMPLETED


async def test_batch_validation(harness: Harness) -> None:
    """Reject empty directories, paths outside the allowlist and oversized batches."""
    with pytest.raises(InvalidRequestError, match="no supported audio"):
        await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
    with pytest.raises(InvalidRequestError, match="not inside"):
        await harness.jobs.submit(BatchTranscribeCommand(directory=Path("/tmp")), owner=OWNER)
    _files(harness, [f"{i}.wav" for i in range(3)])
    limited = build_harness(harness.inputs.parent / "limited", settings_overrides={"jobs": {"batch_max_items": 2}})
    _files(limited, [f"{i}.wav" for i in range(3)])
    with pytest.raises(InvalidRequestError, match="more than 2"):
        await limited.jobs.submit(BatchTranscribeCommand(directory=limited.inputs), owner=OWNER)
    limited.executor.shutdown()


async def test_batch_scan_runs_off_the_event_loop(harness: Harness) -> None:
    """List the batch directory in a worker thread so a large tree does not block the event loop."""
    _files(harness, ["a.wav", "b.wav"])
    policy = harness.planner._input_policy
    threads: list[int] = []
    original = policy.audio_files

    def recording_audio_files(directory: Path, *, recursive: bool, limit: int) -> list[Path]:
        """Record the calling thread and delegate to the real scan."""
        threads.append(threading.get_ident())
        return original(directory, recursive=recursive, limit=limit)

    policy.audio_files = recording_audio_files  # type: ignore[method-assign]
    submission = await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
    assert submission.job_id is not None
    assert threads
    assert threads[0] != threading.get_ident()
