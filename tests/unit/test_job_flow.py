"""Job submission, execution, retries, recovery and ownership through the real services and worker."""

from pathlib import Path

import pytest

from app.domain.errors import (
    AssetPublishError,
    InferenceError,
    InvalidRequestError,
    ModelNotInstalledError,
    NotFoundError,
    PersistenceUnavailableError,
)
from app.domain.jobs import JobFilter, JobStatus, TranscribeCommand
from app.domain.models import ModelKind
from app.domain.stt import SttBackendName, SttOptions
from tests.conftest import write_tone
from tests.harness import OTHER, OWNER, Harness, build_harness
from tests.mocks.storage import MockAssetStore
from tests.mocks.stt import MLX_LIKE_CAPABILITIES, MockSttBackend
from tests.mocks.vad import MockVoiceActivityDetector


def _command(harness: Harness, **kwargs: object) -> TranscribeCommand:
    """Return a transcription command for a fresh input file."""
    path = write_tone(harness.inputs / "speech.wav")
    return TranscribeCommand(input_path=path, **kwargs)  # type: ignore[arg-type]


async def test_submit_commits_then_notifies(harness: Harness) -> None:
    """Persist a queued job and wake the worker after commit."""
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.QUEUED
    assert harness.queue.notified == [submission.job_id]


async def test_job_completes_with_files_transcript_and_metrics(harness: Harness, tmp_path: Path) -> None:
    """Run a job to completion and expose output files, preview text and stored transcript."""
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.COMPLETED
    assert set(view.files) == {"txt", "json", "srt", "vtt"}
    assert view.result["text"] == "hello world"
    assert submission.job_id in harness.uow.db.transcripts
    assert submission.job_id in harness.uow.db.metrics
    assert any(a.asset.name == "original" for a in harness.uow.db.assets)
    assert not (tmp_path / "workspace" / "jobs" / str(submission.job_id)).exists()


async def test_request_id_deduplicates_per_owner(harness: Harness) -> None:
    """Return the existing job for a repeated request_id, but not across owners."""
    first = await harness.jobs.submit(_command(harness, request_id="r1"), owner=OWNER)
    again = await harness.jobs.submit(_command(harness, request_id="r1"), owner=OWNER)
    other = await harness.jobs.submit(_command(harness, request_id="r1"), owner=OTHER)
    assert again.job_id == first.job_id
    assert again.deduplicated
    assert other.job_id != first.job_id


async def test_cross_caller_access_is_not_found(harness: Harness) -> None:
    """Hide other owners' jobs exactly like missing ones."""
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    with pytest.raises(NotFoundError):
        await harness.queries.get(submission.job_id, owner=OTHER)
    page = await harness.queries.list(owner=OTHER, job_filter=JobFilter(), limit=None, cursor=None)
    assert page.items == ()


async def test_list_jobs_pagination_and_filters(harness: Harness) -> None:
    """Page newest first with a cursor and filter by status."""
    ids = []
    for _ in range(3):
        ids.append((await harness.jobs.submit(_command(harness), owner=OWNER)).job_id)
        harness.clock.advance(seconds=1)
    first = await harness.queries.list(owner=OWNER, job_filter=JobFilter(), limit=2, cursor=None)
    second = await harness.queries.list(owner=OWNER, job_filter=JobFilter(), limit=2, cursor=first.next_cursor)
    assert [v.id for v in first.items] == [ids[2], ids[1]]
    assert [v.id for v in second.items] == [ids[0]]
    assert second.next_cursor is None
    completed = await harness.queries.list(
        owner=OWNER, job_filter=JobFilter(status=JobStatus.COMPLETED), limit=None, cursor=None
    )
    assert completed.items == ()
    with pytest.raises(InvalidRequestError):
        await harness.queries.list(owner=OWNER, job_filter=JobFilter(), limit=101, cursor=None)


async def test_wait_returns_terminal_view(harness: Harness) -> None:
    """Return the completed job when it finishes within the wait."""
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    await harness.run_queued()
    view = await harness.jobs.wait_for(submission.job_id, owner=OWNER, timeout_seconds=5)
    assert view.status is JobStatus.COMPLETED


async def test_wait_times_out_with_current_status(harness: Harness) -> None:
    """Return the current status after the timeout without failing the job."""
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    view = await harness.jobs.wait_for(submission.job_id, owner=OWNER, timeout_seconds=0)
    assert view.status is JobStatus.QUEUED


async def test_uninstalled_requested_model_is_rejected(harness: Harness) -> None:
    """Reject a caller-requested model that is not installed."""
    with pytest.raises(ModelNotInstalledError):
        await harness.jobs.submit(_command(harness, model="medium"), owner=OWNER)


async def test_preset_model_is_substituted_and_reported(harness: Harness) -> None:
    """Substitute an uninstalled preset model and surface it as a warning."""
    submission = await harness.jobs.submit(_command(harness, preset="balanced"), owner=OWNER)
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert any("model 'medium'" in warning for warning in view.warnings)


async def test_non_retryable_error_fails_job(tmp_path: Path) -> None:
    """Fail immediately on non-retryable inference errors."""
    harness = build_harness(tmp_path, stt=MockSttBackend(error=InferenceError("oom", retryable=False)))
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.FAILED
    assert view.error is not None
    assert view.error.code == "inference_error"
    harness.executor.shutdown()


async def test_retryable_publish_failure_retries_without_rerunning_inference(tmp_path: Path) -> None:
    """Schedule a retry on publish failure and reuse staged outputs on the next attempt."""
    harness = build_harness(tmp_path, store=MockAssetStore(fail_puts=1))
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.RETRY_WAIT
    assert view.error is not None
    assert view.error.code == AssetPublishError.code
    harness.clock.advance(seconds=5)
    await harness.recovery.recover()
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.COMPLETED
    assert len(harness.stt.requests) == 1
    harness.executor.shutdown()


async def test_retries_exhausted_fail_the_job(tmp_path: Path) -> None:
    """Fail after max_attempts retryable failures."""
    harness = build_harness(tmp_path, store=MockAssetStore(fail_puts=10))
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    for _ in range(3):
        await harness.run_queued()
        harness.clock.advance(seconds=30)
        await harness.recovery.recover()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.FAILED
    harness.executor.shutdown()


async def test_recovery_interrupts_stale_running_jobs(harness: Harness) -> None:
    """Requeue a running job whose heartbeat expired."""
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    await harness.transitions.claim(submission.job_id, lease_owner="dead-worker")
    harness.clock.advance(seconds=600)
    harness.queue.notified.clear()
    await harness.recovery.recover()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.QUEUED
    assert harness.queue.notified == [submission.job_id]


async def test_database_outage_maps_to_persistence_unavailable(harness: Harness) -> None:
    """Fail submissions with persistence_unavailable while the database is down."""
    harness.uow.available = False
    with pytest.raises(PersistenceUnavailableError):
        await harness.jobs.submit(_command(harness), owner=OWNER)


async def test_mlx_path_runs_standalone_vad_and_maps_timestamps(tmp_path: Path) -> None:
    """Run Silero before STT on backends without native VAD and shift timestamps back to source time."""
    vad = MockVoiceActivityDetector()
    stt = MockSttBackend(capabilities=MLX_LIKE_CAPABILITIES, name=SttBackendName.MLX)
    harness = build_harness(tmp_path, stt=stt, vad=vad, installed={(ModelKind.STT, "mlx", "small")}, tts=None)
    submission = await harness.jobs.submit(_command(harness, preset="accurate"), owner=OWNER)
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    transcript = harness.uow.db.transcripts[submission.job_id]
    assert vad.calls == 1
    assert stt.requests[0].vad.enabled is False
    assert stt.requests[0].beam_size == 1
    assert transcript.segments[0].start == pytest.approx(0.5)
    assert any("beam_size" in w for w in view.warnings)
    harness.executor.shutdown()


async def test_mlx_no_speech_skips_model(tmp_path: Path) -> None:
    """Return an empty transcript with no_speech_detected without calling the model."""
    stt = MockSttBackend(capabilities=MLX_LIKE_CAPABILITIES, name=SttBackendName.MLX)
    harness = build_harness(
        tmp_path, stt=stt, vad=MockVoiceActivityDetector(()), installed={(ModelKind.STT, "mlx", "small")}, tts=None
    )
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.COMPLETED
    assert "no_speech_detected" in view.warnings
    assert stt.requests == []
    harness.executor.shutdown()


async def test_explicit_caller_option_unsupported_on_backend(tmp_path: Path) -> None:
    """Reject caller hotwords on a backend without hotword support."""
    stt = MockSttBackend(capabilities=MLX_LIKE_CAPABILITIES, name=SttBackendName.MLX)
    harness = build_harness(tmp_path, stt=stt, installed={(ModelKind.STT, "mlx", "small")}, tts=None)
    with pytest.raises(InvalidRequestError, match="hotwords"):
        await harness.jobs.submit(_command(harness, options=SttOptions(hotwords=("x",))), owner=OWNER)
    harness.executor.shutdown()


async def test_denoise_runs_when_requested(tmp_path: Path) -> None:
    """Denoise before transcription when requested and installed."""
    harness = build_harness(tmp_path, denoiser=True)
    await harness.jobs.submit(_command(harness, options=SttOptions(denoise=True)), owner=OWNER)
    await harness.run_queued()
    assert harness.denoiser is not None
    assert len(harness.denoiser.calls) == 1
    harness.executor.shutdown()


async def test_preset_denoise_is_skipped_with_warning_when_missing(harness: Harness) -> None:
    """Skip preset denoising without the extra and record a warning."""
    submission = await harness.jobs.submit(_command(harness, preset="studio"), owner=OWNER)
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert any("denoise" in w for w in view.warnings)


async def test_cli_retry_and_cancel_transitions(harness: Harness) -> None:
    """Allow cancel from queued and retry from failed only."""
    submission = await harness.jobs.submit(_command(harness), owner=OWNER)
    async with harness.uow() as uow:
        assert await uow.jobs.set_status(submission.job_id, JobStatus.CANCELLED, now=harness.clock.now())
        assert not await uow.jobs.set_status(submission.job_id, JobStatus.QUEUED, now=harness.clock.now())
