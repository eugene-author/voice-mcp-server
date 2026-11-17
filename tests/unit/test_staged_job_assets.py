"""Staged retry checkpoints: the StagedJobAssets mechanics and STT/TTS retries through the real services."""

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from app.domain.ids import JobId, new_claim_token, new_job_id
from app.domain.jobs import ClaimedJob, JobStatus, JobType, SynthesizeCommand, TranscribeCommand
from app.services.job_workspace import JobWorkspace
from app.services.staged_job_assets import COMPLETE_MARKER, METADATA_FILE, StagedJobAssets
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness
from tests.mocks.storage import MockAssetStore


def _metadata(_directory: Path, metadata: Mapping[str, Any]) -> int:
    """Parse a checkpoint whose metadata must carry an integer count."""
    return int(metadata["count"])


@pytest.fixture
def staged(tmp_path: Path) -> StagedJobAssets:
    """Return staged assets over a temporary workspace root."""
    return StagedJobAssets(JobWorkspace(tmp_path / "workspace"))


def _claim(job_id: JobId | None = None) -> ClaimedJob:
    """Return a new claim, of a new job unless one is given."""
    return ClaimedJob(
        id=job_id or new_job_id(),
        type=JobType.SPEECH_TO_TEXT,
        owner=OWNER,
        input_payload={},
        config_snapshot={},
        attempt_count=1,
        max_attempts=2,
        claim_token=new_claim_token(),
    )


def _commit(staged: StagedJobAssets, claim: ClaimedJob, text: str = "hello", count: int = 3) -> None:
    """Commit one output with the given content."""
    staged.write_text(claim, "out.txt", text)
    staged.commit(claim, {"count": count}, required=("out.txt",))


def test_later_claim_reuses_a_committed_checkpoint_in_its_own_directory(staged: StagedJobAssets) -> None:
    """Copy outputs, metadata and marker into the new claim, leaving the earlier claim's directory untouched."""
    first = _claim()
    _commit(staged, first)
    second = _claim(first.id)
    assert staged.load(second, required=("out.txt",), parse=_metadata) == 3
    assert staged.path(second, "out.txt").parent != staged.path(first, "out.txt").parent
    assert staged.path(second, "out.txt").read_text(encoding="utf-8") == "hello"
    assert staged.path(second, COMPLETE_MARKER).exists()
    assert not staged.path(second, "out.txt.part").exists()
    assert staged.path(first, COMPLETE_MARKER).exists()


def test_writes_of_two_claims_never_touch_each_other(staged: StagedJobAssets) -> None:
    """Keep each claim's content when both write the same file names."""
    first = _claim()
    second = _claim(first.id)
    _commit(staged, second, text="current")
    _commit(staged, first, text="stale")
    assert staged.path(second, "out.txt").read_text(encoding="utf-8") == "current"


def _two_commits(staged: StagedJobAssets) -> tuple[ClaimedJob, ClaimedJob]:
    """Return two claims of one job with committed checkpoints, the second newer, counting 1 and 2."""
    older = _claim()
    newer = _claim(older.id)
    for claim, count, stamp in ((older, 1, 1_000), (newer, 2, 2_000)):
        _commit(staged, claim, count=count)
        os.utime(staged.path(claim, COMPLETE_MARKER), ns=(stamp, stamp))
    return older, newer


def test_newest_commit_wins(staged: StagedJobAssets) -> None:
    """Prefer the newest of several committed checkpoints."""
    older, _ = _two_commits(staged)
    assert staged.load(_claim(older.id), required=("out.txt",), parse=_metadata) == 2


def test_damaged_commit_is_skipped_without_changing_it(staged: StagedJobAssets) -> None:
    """Fall back past a damaged checkpoint and leave that claim's files alone."""
    older, newer = _two_commits(staged)
    staged.path(newer, METADATA_FILE).write_text("{", encoding="utf-8")
    assert staged.load(_claim(older.id), required=("out.txt",), parse=_metadata) == 1
    assert staged.path(newer, METADATA_FILE).read_text(encoding="utf-8") == "{"


def test_commit_refuses_missing_outputs(staged: StagedJobAssets) -> None:
    """Keep a checkpoint incomplete while a required output is missing."""
    claim = _claim()
    with pytest.raises(FileNotFoundError):
        staged.commit(claim, {"count": 1}, required=("out.txt",))
    assert not staged.path(claim, COMPLETE_MARKER).exists()


def test_outputs_without_marker_are_ignored(staged: StagedJobAssets) -> None:
    """Ignore files of a claim that crashed before committing, without touching that claim's directory."""
    crashed = _claim()
    staged.write_text(crashed, "out.txt", "partial")
    staged.write_text(crashed, METADATA_FILE, json.dumps({"count": 1}))
    retry = _claim(crashed.id)
    assert staged.load(retry, required=("out.txt",), parse=_metadata) is None
    assert list(staged.path(retry, "x").parent.iterdir()) == []
    assert staged.path(crashed, "out.txt").exists()


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: (d / METADATA_FILE).write_text('{"count": ', encoding="utf-8"),
        lambda d: (d / METADATA_FILE).write_text("[1, 2]", encoding="utf-8"),
        lambda d: (d / METADATA_FILE).write_text('{"other": 1}', encoding="utf-8"),
        lambda d: (d / METADATA_FILE).unlink(),
        lambda d: (d / "out.txt").unlink(),
    ],
    ids=["truncated-json", "not-an-object", "missing-field", "missing-metadata", "missing-output"],
)
def test_unusable_checkpoint_is_not_reused(staged: StagedJobAssets, damage: Callable[[Path], object]) -> None:
    """Return None for a damaged checkpoint and leave the new claim's directory empty, so it regenerates outputs."""
    first = _claim()
    _commit(staged, first)
    damage(staged.path(first, "out.txt").parent)
    retry = _claim(first.id)
    assert staged.load(retry, required=("out.txt",), parse=_metadata) is None
    assert list(staged.path(retry, "x").parent.iterdir()) == []


def _only_claim(harness: Harness, job_id: JobId) -> Path:
    """Return the one claim directory of a job."""
    [directory] = (harness.workspace.job_dir(job_id) / "claims").iterdir()
    return directory


async def _fail_first_publish(harness: Harness, command: object) -> Path:
    """Submit, run once against a failing store, and return the job's staged directory."""
    submission = await harness.jobs.submit(command, owner=OWNER)  # type: ignore[arg-type]
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.RETRY_WAIT
    harness.extra["job_id"] = submission.job_id
    return _only_claim(harness, submission.job_id) / "staged"


async def _retry(harness: Harness) -> JobStatus:
    """Let the retry become due, run it, and return the job's status."""
    harness.clock.advance(seconds=5)
    await harness.recovery.recover()
    await harness.run_queued()
    return (await harness.queries.get(harness.extra["job_id"], owner=OWNER)).status


async def test_stt_retry_reuses_checkpoint_without_inference(tmp_path: Path) -> None:
    """Publish the staged transcript on retry without transcribing again."""
    harness = build_harness(tmp_path, store=MockAssetStore(fail_puts=1))
    await _fail_first_publish(harness, TranscribeCommand(input_path=write_tone(harness.inputs / "a.wav")))
    assert await _retry(harness) is JobStatus.COMPLETED
    assert len(harness.stt.requests) == 1
    harness.executor.shutdown()


@pytest.mark.parametrize("damage", ["truncate-metadata", "drop-transcript", "corrupt-transcript-json"])
async def test_stt_retry_regenerates_damaged_checkpoint(tmp_path: Path, damage: str) -> None:
    """Transcribe again, instead of failing on a traceback, when the staged checkpoint is damaged."""
    harness = build_harness(tmp_path, store=MockAssetStore(fail_puts=1))
    directory = await _fail_first_publish(harness, TranscribeCommand(input_path=write_tone(harness.inputs / "a.wav")))
    match damage:
        case "truncate-metadata":
            (directory / METADATA_FILE).write_text('{"warnings": [', encoding="utf-8")
        case "drop-transcript":
            (directory / "transcript.srt").unlink()
        case "corrupt-transcript-json":
            (directory / "transcript.json").write_text('{"segments": 7}', encoding="utf-8")
    assert await _retry(harness) is JobStatus.COMPLETED
    assert len(harness.stt.requests) == 2
    harness.executor.shutdown()


async def test_tts_retry_reuses_checkpoint_without_synthesis(tmp_path: Path) -> None:
    """Publish the staged audio on retry without synthesizing again."""
    harness = build_harness(tmp_path, store=MockAssetStore(fail_puts=1))
    await _fail_first_publish(harness, SynthesizeCommand(text="Hello there.", voice_id="preset:vivian"))
    assert harness.tts is not None
    calls = len(harness.tts.requests)
    assert await _retry(harness) is JobStatus.COMPLETED
    assert len(harness.tts.requests) == calls
    harness.executor.shutdown()


@pytest.mark.parametrize("damage", ["truncate-metadata", "bad-sample-rate", "drop-audio"])
async def test_tts_retry_regenerates_damaged_checkpoint(tmp_path: Path, damage: str) -> None:
    """Synthesize again when the staged metadata is unreadable or the audio file is gone."""
    harness = build_harness(tmp_path, store=MockAssetStore(fail_puts=1))
    directory = await _fail_first_publish(harness, SynthesizeCommand(text="Hello there.", voice_id="preset:vivian"))
    assert harness.tts is not None
    calls = len(harness.tts.requests)
    match damage:
        case "truncate-metadata":
            (directory / METADATA_FILE).write_text("{", encoding="utf-8")
        case "bad-sample-rate":
            metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
            (directory / METADATA_FILE).write_text(json.dumps({**metadata, "sample_rate": "x"}), encoding="utf-8")
        case "drop-audio":
            next(directory.glob("speech.*")).unlink()
    assert await _retry(harness) is JobStatus.COMPLETED
    assert len(harness.tts.requests) == 2 * calls
    harness.executor.shutdown()
