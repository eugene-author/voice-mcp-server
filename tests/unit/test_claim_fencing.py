"""Claim fencing: a worker that lost its claim can change neither the job nor the newer claim's files or outputs."""

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from app.domain.assets import AssetRole
from app.domain.errors import ClaimLostError, InferenceError, InvalidRequestError
from app.domain.ids import JobId
from app.domain.jobs import (
    BatchTranscribeCommand,
    ClaimedJob,
    JobCompletion,
    JobPhase,
    JobStatus,
    JobType,
    RegisterVoiceCommand,
    TranscribeCommand,
)
from app.domain.voices import VoiceStatus
from app.workers.heartbeat import Heartbeat
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness
from tests.mocks.handlers import MockBlockingHandler, MockScriptedHandler
from tests.mocks.media import MockMedia

SAME_PROCESS = "gpu-host:1"
LEASE_EXPIRED_SECONDS = 121


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Return a harness whose voice references are long enough to register."""
    harness = build_harness(tmp_path, media=MockMedia(duration_seconds=8))
    yield harness
    harness.executor.shutdown()


async def _submit(harness: Harness) -> JobId:
    """Submit a transcription job."""
    path = write_tone(harness.inputs / "a.wav", seconds=8)
    return (await harness.jobs.submit(TranscribeCommand(input_path=path), owner=OWNER)).job_id


async def _claim(harness: Harness, job_id: JobId) -> ClaimedJob:
    """Claim a job that must be claimable, always from the same process."""
    claim = await harness.transitions.claim(job_id, lease_owner=SAME_PROCESS)
    assert claim is not None
    return claim


async def _reclaim(harness: Harness, job_id: JobId) -> ClaimedJob:
    """Expire the current lease, let recovery requeue the job, and claim it again."""
    harness.clock.advance(seconds=LEASE_EXPIRED_SECONDS)
    await harness.recovery.recover()
    return await _claim(harness, job_id)


async def _lost_claim(harness: Harness) -> tuple[ClaimedJob, ClaimedJob]:
    """Return a claim that lost its job and the newer claim, both from the same process."""
    stale = await _claim(harness, await _submit(harness))
    return stale, await _reclaim(harness, stale.id)


def _events(harness: Harness, job_id: JobId) -> list[str]:
    """Return the job's event types in order."""
    return [event.event_type for event in harness.uow.db.events if event.job_id == job_id]


def _worker_steals(
    harness: Harness,
    *,
    after_steal: Callable[[], None],
    completion: JobCompletion | None = None,
    error: Exception | None = None,
) -> None:
    """Make the next transcription lose its claim to a newer claim while it runs, then finish as given."""

    async def steal(job: ClaimedJob) -> None:
        """Hand the job to a newer claim, remembered for the test."""
        harness.extra["current"] = await _reclaim(harness, job.id)
        after_steal()

    harness.handlers[JobType.SPEECH_TO_TEXT] = MockScriptedHandler(during=steal, completion=completion, error=error)


async def test_claims_of_one_process_get_different_tokens(harness: Harness) -> None:
    """Tell apart two claims with the same lease owner string."""
    stale, current = await _lost_claim(harness)
    assert harness.uow.db.jobs[stale.id].lease_owner == SAME_PROCESS
    assert stale.claim_token != current.claim_token


async def test_stale_completion_changes_nothing(harness: Harness) -> None:
    """Reject the old completion, leaving the job, transcript, assets, metrics and events to the newer claim."""
    stale, current = await _lost_claim(harness)
    before = _events(harness, stale.id)
    assert (
        await harness.transitions.mark_completed(stale, JobCompletion(result={"from": "stale"}, metrics={"x": 1}))
        is None
    )
    record = harness.uow.db.jobs[stale.id]
    assert (record.status, record.result, record.claim_token) == (JobStatus.RUNNING, {}, current.claim_token)
    assert harness.uow.db.assets == []
    assert stale.id not in harness.uow.db.metrics
    assert stale.id not in harness.uow.db.transcripts
    assert _events(harness, stale.id) == before
    assert (
        await harness.transitions.mark_completed(current, JobCompletion(result={"from": "current"}))
        is JobStatus.COMPLETED
    )
    assert harness.uow.db.jobs[stale.id].result == {"from": "current"}


async def test_stale_failure_changes_nothing(harness: Harness) -> None:
    """Reject the old failure, without failing the job."""
    stale, _ = await _lost_claim(harness)
    before = _events(harness, stale.id)
    assert await harness.transitions.mark_failed(stale, InvalidRequestError("stale")) is None
    assert harness.uow.db.jobs[stale.id].status is JobStatus.RUNNING
    assert _events(harness, stale.id) == before


async def test_stale_retry_is_not_reported_as_scheduled(harness: Harness) -> None:
    """Honor the rejected update: no retry_wait, no retry event, no retry reported."""
    stale, _ = await _lost_claim(harness)
    assert await harness.transitions.schedule_retry(stale, InferenceError("stale", retryable=True)) is None
    assert harness.uow.db.jobs[stale.id].status is JobStatus.RUNNING
    assert "retry_scheduled" not in _events(harness, stale.id)


async def test_stale_progress_stops_the_pipeline_and_records_nothing(harness: Harness) -> None:
    """Raise ClaimLostError instead of moving the newer claim's phase or adding an event."""
    stale, current = await _lost_claim(harness)
    await harness.transitions.progress_for(current)(JobPhase.DECODING, 5)
    before = _events(harness, stale.id)
    with pytest.raises(ClaimLostError):
        await harness.transitions.progress_for(stale)(JobPhase.PUBLISHING, 92)
    record = harness.uow.db.jobs[stale.id]
    assert (record.phase, record.progress) == (JobPhase.DECODING, 5)
    assert _events(harness, stale.id) == before


async def test_stale_heartbeat_is_rejected(harness: Harness) -> None:
    """Report the claim as lost and leave the newer claim's lease alone."""
    stale, current = await _lost_claim(harness)
    heartbeat = harness.uow.db.jobs[stale.id].heartbeat_at
    harness.clock.advance(seconds=1)
    beats = Heartbeat(uow=harness.uow, clock=harness.clock, interval_seconds=60)
    assert not await beats.beat(stale)
    assert harness.uow.db.jobs[stale.id].heartbeat_at == heartbeat
    assert await beats.beat(current)


async def test_stale_release_is_rejected(harness: Harness) -> None:
    """Keep the newer claim running with its attempt counted."""
    stale, current = await _lost_claim(harness)
    assert not await harness.transitions.release(stale)
    record = harness.uow.db.jobs[stale.id]
    assert (record.status, record.attempt_count, record.claim_token) == (JobStatus.RUNNING, 2, current.claim_token)
    assert "released" not in _events(harness, stale.id)


async def test_refunded_attempt_still_gets_a_new_token(harness: Harness) -> None:
    """Invalidate the released claim even though the reclaim repeats its attempt count and owner."""
    released = await _claim(harness, await _submit(harness))
    assert await harness.transitions.release(released)
    again = await _claim(harness, released.id)
    assert again.attempt_count == released.attempt_count
    assert again.claim_token != released.claim_token
    assert await harness.transitions.mark_completed(released, JobCompletion()) is None
    assert harness.uow.db.jobs[released.id].status is JobStatus.RUNNING


async def test_stale_registration_leaves_the_voice_pending(harness: Harness) -> None:
    """Change no voice, and record no asset, when the claim that registered it was lost."""
    reference = write_tone(harness.inputs / "me.wav", seconds=8)
    command = RegisterVoiceCommand(
        name="me", reference_audio_path=reference, reference_text="hello world", consent_confirmed=True
    )
    submission = await harness.jobs.submit(command, owner=OWNER)
    assert submission.voice_id is not None
    stale = await _claim(harness, submission.job_id)
    completion = await harness.handlers[JobType.REGISTER_VOICE].execute(stale, _no_progress)
    await _reclaim(harness, stale.id)
    assert await harness.transitions.mark_completed(stale, completion) is None
    assert await harness.transitions.mark_failed(stale, InvalidRequestError("stale")) is None
    assert harness.uow.db.voices[submission.voice_id].status is VoiceStatus.PENDING
    assert harness.uow.db.assets == []


async def _no_progress(_phase: JobPhase, _progress: int) -> None:
    """Report nothing, as a pipeline past its last progress write does."""


async def _publish_as(harness: Harness, claim: ClaimedJob, text: str) -> JobCompletion:
    """Run the real transcription handler for a claim, with its transcript text."""
    harness.stt._text = text
    return await harness.handlers[JobType.SPEECH_TO_TEXT].execute(claim, _no_progress)


async def test_late_upload_cannot_overwrite_the_accepted_outputs(harness: Harness) -> None:
    """Keep the accepted transcript when the stale claim publishes after the job completed."""
    stale, current = await _lost_claim(harness)
    accepted = await _publish_as(harness, current, "current words")
    assert await harness.transitions.mark_completed(current, accepted) is JobStatus.COMPLETED
    late = await _publish_as(harness, stale, "stale words")
    assert await harness.transitions.mark_completed(stale, late) is None
    recorded = {record.asset.name: record.asset for record in harness.uow.db.assets}
    assert harness.store.blobs[recorded["txt"].stored.location.uri].strip() == b"current words"
    assert {asset.stored.location.uri for asset in late.assets}.isdisjoint(
        asset.stored.location.uri for asset in recorded.values()
    )
    assert all(str(current.claim_token) in asset.key.value for asset in recorded.values())


async def test_preserved_original_is_scoped_to_the_claim(harness: Harness) -> None:
    """Upload a shared-path original under the claim, so a stale upload cannot replace the recorded object."""
    stale, current = await _lost_claim(harness)
    accepted = await _publish_as(harness, current, "current words")
    late = await _publish_as(harness, stale, "stale words")
    originals = [
        next(asset for asset in completion.assets if asset.role is AssetRole.INPUT) for completion in (accepted, late)
    ]
    assert originals[0].stored.location != originals[1].stored.location


async def test_late_voice_reference_cannot_replace_the_accepted_one(harness: Harness) -> None:
    """Point the voice at the accepted claim's recording, and leave the stale upload unrecorded for reconciliation."""
    reference = write_tone(harness.inputs / "me.wav", seconds=8)
    command = RegisterVoiceCommand(
        name="me", reference_audio_path=reference, reference_text="hello world", consent_confirmed=True
    )
    submission = await harness.jobs.submit(command, owner=OWNER)
    handler = harness.handlers[JobType.REGISTER_VOICE]
    stale = await _claim(harness, submission.job_id)
    current = await _reclaim(harness, submission.job_id)
    accepted = await handler.execute(current, _no_progress)
    assert await harness.transitions.mark_completed(current, accepted) is JobStatus.COMPLETED
    late = await handler.execute(stale, _no_progress)
    assert await harness.transitions.mark_completed(stale, late) is None
    assert late.voice_reference.stored.location != accepted.voice_reference.stored.location
    recorded = {record.asset.stored.location for record in harness.uow.db.assets}
    assert accepted.voice_reference.stored.location in recorded
    assert late.voice_reference.stored.location not in recorded


async def test_uploaded_input_is_never_rewritten_by_a_claim(tmp_path: Path) -> None:
    """Keep the one uploaded object for every claim, with only the outputs scoped."""
    harness = build_harness(tmp_path, settings_overrides={"storage": {"input_transfer": "ingest"}})
    try:
        stale, current = await _lost_claim(harness)
        completions = [await _publish_as(harness, claim, "words") for claim in (current, stale)]
        originals = {
            asset.stored.location
            for completion in completions
            for asset in completion.assets
            if asset.role is AssetRole.INPUT
        }
        assert len(originals) == 1
        assert sum("/inputs/" in uri for uri in harness.store.blobs) == 1
    finally:
        harness.executor.shutdown()


def test_cleanup_of_one_claim_keeps_the_other(harness: Harness) -> None:
    """Remove only the stale claim's directory."""
    jobs = asyncio.run(_two_claim_dirs(harness))
    stale, current = jobs
    harness.workspace.remove_claim(stale)
    assert not (harness.workspace.job_dir(stale.id) / "claims" / str(stale.claim_token)).exists()
    assert (harness.workspace.claim_dir(current) / "keep.txt").read_text(encoding="utf-8") == "current"


async def _two_claim_dirs(harness: Harness) -> tuple[ClaimedJob, ClaimedJob]:
    """Return a lost claim and the newer claim, each with a file in its directory."""
    stale, current = await _lost_claim(harness)
    (harness.workspace.claim_dir(stale) / "keep.txt").write_text("stale", encoding="utf-8")
    (harness.workspace.claim_dir(current) / "keep.txt").write_text("current", encoding="utf-8")
    return stale, current


def test_removing_the_last_claim_leaves_no_job_directory(harness: Harness) -> None:
    """Prune the empty job directories together with the last claim."""
    stale, current = asyncio.run(_two_claim_dirs(harness))
    harness.workspace.remove_claim(stale)
    harness.workspace.remove_claim(current)
    assert not (harness.workspace._root / "jobs" / str(stale.id)).exists()


async def test_keep_alive_reports_a_rejection_once_and_stops(harness: Harness) -> None:
    """Call on_lost when the claim is rejected and end the loop."""
    stale, _ = await _lost_claim(harness)
    lost: list[bool] = []
    beats = Heartbeat(uow=harness.uow, clock=harness.clock, interval_seconds=0.001)
    await asyncio.wait_for(beats.keep_alive(stale, on_lost=lambda: lost.append(True)), 5)
    assert lost == [True]


async def test_keep_alive_survives_a_database_failure(harness: Harness) -> None:
    """Treat an unavailable database as transient, not as losing the claim."""
    claim = await _claim(harness, await _submit(harness))
    lost: list[bool] = []
    harness.uow.available = False
    beats = Heartbeat(uow=harness.uow, clock=harness.clock, interval_seconds=0.001)
    task = asyncio.create_task(beats.keep_alive(claim, on_lost=lambda: lost.append(True)))
    await asyncio.sleep(0.02)
    harness.uow.available = True
    await asyncio.sleep(0.02)
    assert not task.done()
    assert lost == []
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_rejected_heartbeat_stops_the_running_handler(tmp_path: Path) -> None:
    """Cancel the handler, record nothing for it, and leave the newer claim's files alone."""
    harness = build_harness(tmp_path)
    blocked = MockBlockingHandler(harness.handlers[JobType.SPEECH_TO_TEXT])
    harness.handlers[JobType.SPEECH_TO_TEXT] = blocked
    harness.worker._heartbeat = Heartbeat(uow=harness.uow, clock=harness.clock, interval_seconds=0.005)
    job_id = await _submit(harness)
    run = asyncio.create_task(harness.run_queued())
    await asyncio.wait_for(blocked.started.wait(), 5)
    current = await _reclaim(harness, job_id)
    (harness.workspace.claim_dir(current) / "keep.txt").write_text("current", encoding="utf-8")
    await asyncio.wait_for(run, 5)
    assert blocked.cancelled
    record = harness.uow.db.jobs[job_id]
    assert (record.status, record.claim_token) == (JobStatus.RUNNING, current.claim_token)
    assert (harness.workspace.claim_dir(current) / "keep.txt").exists()
    harness.executor.shutdown()


async def test_rejected_progress_stops_the_handler_without_failing_the_job(tmp_path: Path) -> None:
    """Stop at the pipeline's next progress write, without reporting a failure for the newer claim's job."""
    harness = build_harness(tmp_path)
    blocked = MockBlockingHandler(harness.handlers[JobType.SPEECH_TO_TEXT])
    harness.handlers[JobType.SPEECH_TO_TEXT] = blocked
    job_id = await _submit(harness)
    run = asyncio.create_task(harness.run_queued())
    await asyncio.wait_for(blocked.started.wait(), 5)
    current = await _reclaim(harness, job_id)
    blocked.finish.set()
    await asyncio.wait_for(run, 5)
    record = harness.uow.db.jobs[job_id]
    assert (record.status, record.error, record.claim_token) == (JobStatus.RUNNING, None, current.claim_token)
    assert harness.stt.requests == []
    harness.executor.shutdown()


@pytest.mark.parametrize(
    "finish",
    [
        {"completion": JobCompletion(result={"from": "stale"})},
        {"error": InvalidRequestError("stale failure")},
        {"error": InferenceError("stale failure", retryable=True)},
    ],
    ids=["completion", "failure", "retry"],
)
async def test_worker_drops_a_rejected_result_and_only_its_own_files(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, finish: dict[str, object]
) -> None:
    """Resolve no waiter, aggregate no batch, and clean only the stale claim's workspace."""
    resolved: list[JobId] = []
    aggregated: list[JobId] = []

    async def aggregate(parent_id: JobId) -> None:
        """Record an aggregation."""
        aggregated.append(parent_id)

    monkeypatch.setattr(harness.waiters, "resolve_if_terminal", lambda job_id, _status: resolved.append(job_id))
    monkeypatch.setattr(harness.batches, "aggregate", aggregate)
    write_tone(harness.inputs / "a.wav")
    batch = await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs), owner=OWNER)
    [child] = [r.job.id for r in harness.uow.db.jobs.values() if r.job.parent_job_id == batch.job_id]
    _worker_steals(harness, after_steal=aggregated.clear, **finish)  # type: ignore[arg-type]
    await harness.run_queued()
    current = harness.extra["current"]
    record = harness.uow.db.jobs[child]
    assert (record.status, record.claim_token) == (JobStatus.RUNNING, current.claim_token)
    assert (resolved, aggregated) == ([], [])
    assert [path.name for path in harness.workspace.other_claim_dirs(current)] == []


async def test_publish_retry_reuses_the_checkpoint_under_its_own_claim(tmp_path: Path) -> None:
    """Skip inference on the retry, write into the new claim, and publish under the new claim's keys."""
    harness = build_harness(tmp_path)
    original_put = harness.store.put
    tokens: list[str] = []

    async def failing_put(source: Path, *, key: object, content_type: str) -> object:
        """Fail the first attempt's first upload, and record which claim directory each upload reads."""
        tokens.append(source.parts[-3])
        if len(tokens) == 1:
            raise InferenceError("upload failed", retryable=True)
        return await original_put(source, key=key, content_type=content_type)  # type: ignore[arg-type]

    harness.store.put = failing_put  # type: ignore[assignment,method-assign]
    job_id = await _submit(harness)
    await harness.run_queued()
    assert harness.uow.db.jobs[job_id].status is JobStatus.RETRY_WAIT
    harness.clock.advance(seconds=5)
    await harness.recovery.recover()
    await harness.run_queued()
    record = harness.uow.db.jobs[job_id]
    assert record.status is JobStatus.COMPLETED
    assert len(harness.stt.requests) == 1
    assert tokens[0] != tokens[1]
    assert not (harness.workspace._root / "jobs" / str(job_id)).exists()
    harness.executor.shutdown()
