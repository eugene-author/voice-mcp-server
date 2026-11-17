"""Registration job and voice state consistency, and voice deletion, through the real services and worker."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from app.domain.assets import AssetKey, AssetRole, PublishedAsset
from app.domain.errors import AssetPublishError, InvalidRequestError, NotFoundError
from app.domain.ids import JobId, VoiceId, VoiceVersionId, new_voice_version_id
from app.domain.jobs import JobStatus, JobType, RegisterVoiceCommand, TranscribeCommand
from app.domain.voices import VoiceStatus
from tests.conftest import write_tone
from tests.harness import OTHER, OWNER, Harness, build_harness
from tests.mocks.media import MockMedia
from tests.mocks.repositories import AssetRecord
from tests.mocks.stt import MockSttBackend

REFERENCE_TEXT = "Hello, it's nice to meet you today."


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Return a harness whose STT hears the reference text and whose media reports an 8 s clip."""
    harness = build_harness(tmp_path, stt=MockSttBackend(text=REFERENCE_TEXT), media=MockMedia(duration_seconds=8.0))
    yield harness
    harness.executor.shutdown()


async def _submit(harness: Harness, *, owner: object = OWNER, name: str = "reference.wav") -> tuple[JobId, VoiceId]:
    """Submit a registration without running it."""
    path = write_tone(harness.inputs / name, seconds=8.0, sample_rate=24000)
    command = RegisterVoiceCommand(
        name="Narrator", reference_audio_path=path, consent_confirmed=True, reference_text=REFERENCE_TEXT, language="en"
    )
    submission = await harness.jobs.submit(command, owner=owner)  # type: ignore[arg-type]
    assert submission.voice_id is not None
    return submission.job_id, submission.voice_id


def _state(harness: Harness, job_id: JobId, voice_id: VoiceId) -> tuple[JobStatus, VoiceStatus]:
    """Return the stored job and voice status."""
    return harness.uow.db.jobs[job_id].status, harness.uow.db.voices[voice_id].status


def _hear(harness: Harness, text: str) -> None:
    """Change what the mock STT backend hears in the reference recording."""
    harness.stt._text = text


async def _interrupt(harness: Harness, job_id: JobId) -> None:
    """Claim the job as a worker that then dies, and let recovery handle the expired lease."""
    await harness.transitions.claim(job_id, lease_owner="dead-worker")
    harness.clock.advance(seconds=600)
    harness.queue.notified.clear()
    await harness.recovery.recover()


async def test_exhausted_interrupted_registration_fails_its_voice(harness: Harness) -> None:
    """Fail both the job and its pending voice when recovery runs out of attempts."""
    job_id, voice_id = await _submit(harness)
    await _interrupt(harness, job_id)
    assert _state(harness, job_id, voice_id) == (JobStatus.QUEUED, VoiceStatus.PENDING)
    await _interrupt(harness, job_id)
    assert _state(harness, job_id, voice_id) == (JobStatus.FAILED, VoiceStatus.FAILED)


async def test_recoverable_interruption_keeps_voice_pending_and_finishes(harness: Harness) -> None:
    """Requeue an interrupted registration with attempts left; it then completes and the voice becomes ready."""
    job_id, voice_id = await _submit(harness)
    await _interrupt(harness, job_id)
    assert _state(harness, job_id, voice_id) == (JobStatus.QUEUED, VoiceStatus.PENDING)
    await harness.run_queued()
    assert _state(harness, job_id, voice_id) == (JobStatus.COMPLETED, VoiceStatus.READY)


async def test_manual_retry_of_failed_registration_makes_voice_ready(harness: Harness) -> None:
    """Reset the failed voice to pending on retry, then make it ready when the retry succeeds."""
    _hear(harness, "something completely different was said here")
    job_id, voice_id = await _submit(harness)
    await harness.run_queued()
    assert _state(harness, job_id, voice_id) == (JobStatus.FAILED, VoiceStatus.FAILED)

    _hear(harness, REFERENCE_TEXT)
    assert await harness.transitions.retry(job_id)
    assert _state(harness, job_id, voice_id) == (JobStatus.QUEUED, VoiceStatus.PENDING)
    await harness.recovery.wake_queued()
    await harness.run_queued()
    assert _state(harness, job_id, voice_id) == (JobStatus.COMPLETED, VoiceStatus.READY)
    assert harness.uow.db.voices[voice_id].current_version.reference_uri is not None  # type: ignore[union-attr]


async def test_failure_after_manual_retry_fails_both_again(harness: Harness) -> None:
    """Leave the job and voice in consistent failure states when the retried registration fails again."""
    _hear(harness, "something completely different was said here")
    job_id, voice_id = await _submit(harness)
    await harness.run_queued()
    assert await harness.transitions.retry(job_id)
    await harness.recovery.wake_queued()
    await harness.run_queued()
    assert _state(harness, job_id, voice_id) == (JobStatus.FAILED, VoiceStatus.FAILED)


async def test_cancelled_registration_fails_its_voice(harness: Harness) -> None:
    """Do not leave a voice pending forever when its queued registration is cancelled."""
    job_id, voice_id = await _submit(harness)
    assert await harness.transitions.cancel(job_id)
    assert _state(harness, job_id, voice_id) == (JobStatus.CANCELLED, VoiceStatus.FAILED)
    assert not await harness.transitions.retry(job_id)


async def test_pending_voice_cannot_be_deleted(harness: Harness) -> None:
    """Refuse deleting a voice whose registration has not finished."""
    _job_id, voice_id = await _submit(harness)
    with pytest.raises(InvalidRequestError):
        await harness.voices.delete_voice(voice_id, owner=OWNER)
    assert harness.uow.db.voices[voice_id].status is VoiceStatus.PENDING


async def test_retry_is_refused_for_deleted_voice(harness: Harness) -> None:
    """Keep a deleted voice deleted, with cleared references, when its failed registration is retried."""
    _hear(harness, "something completely different was said here")
    job_id, voice_id = await _submit(harness)
    await harness.run_queued()
    await harness.voices.delete_voice(voice_id, owner=OWNER)
    assert not await harness.transitions.retry(job_id)
    assert _state(harness, job_id, voice_id) == (JobStatus.FAILED, VoiceStatus.DELETED)


async def test_late_completion_does_not_revive_deleted_voice(harness: Harness) -> None:
    """Fail a registration that finishes after its voice was deleted and record none of its assets."""
    job_id, voice_id = await _submit(harness)
    job = await harness.transitions.claim(job_id, lease_owner="worker")
    assert job is not None
    completion = await harness.handlers[JobType.REGISTER_VOICE].execute(job, harness.transitions.progress_for(job))
    voice = harness.uow.db.voices[voice_id]
    harness.uow.db.voices[voice_id] = replace(voice, status=VoiceStatus.DELETED)

    status = await harness.transitions.mark_completed(job, completion)

    assert status is JobStatus.FAILED
    assert _state(harness, job_id, voice_id) == (JobStatus.FAILED, VoiceStatus.DELETED)
    assert harness.uow.db.assets == []
    version = harness.uow.db.versions[completion.voice_version_id]  # type: ignore[index]
    assert version.reference_uri is None


async def test_other_job_types_never_touch_voices(harness: Harness) -> None:
    """Leave voice states alone when transcription jobs fail, retry or are cancelled."""
    _registration, voice_id = await _submit(harness)
    audio = write_tone(harness.inputs / "speech.wav", seconds=2.0)
    submission = await harness.jobs.submit(TranscribeCommand(input_path=audio), owner=OWNER)
    assert await harness.transitions.cancel(submission.job_id)
    assert harness.uow.db.voices[voice_id].status is VoiceStatus.PENDING


async def test_delete_retries_after_partial_storage_failure(harness: Harness) -> None:
    """Finish the purge on a second call after storage fails midway through the first."""
    job_id, voice_id = await _submit(harness)
    await harness.run_queued()
    uris = [a.asset.stored.location.uri for a in harness.uow.db.assets]
    assert len(uris) == 2
    harness.store.fail_deletes_after = 1

    with pytest.raises(AssetPublishError):
        await harness.voices.delete_voice(voice_id, owner=OWNER)

    assert harness.uow.db.voices[voice_id].status is VoiceStatus.DELETED
    assert len(harness.uow.db.assets) == 2
    harness.store.fail_deletes_after = None
    assert await harness.voices.delete_voice(voice_id, owner=OWNER) == 2
    assert harness.uow.db.assets == []
    assert all(uri not in harness.store.blobs for uri in uris)
    assert harness.uow.db.jobs[job_id].status is JobStatus.COMPLETED


async def test_multi_version_delete_purges_every_version(harness: Harness) -> None:
    """Purge recordings attached to every version of the voice."""
    _job_id, voice_id = await _submit(harness)
    await harness.run_queued()
    first = harness.uow.db.voices[voice_id].current_version
    assert first is not None
    second_id = await _add_version(harness, voice_id, version=2)
    uris = [a.asset.stored.location.uri for a in harness.uow.db.assets]
    assert {a.voice_version_id for a in harness.uow.db.assets} == {first.id, second_id}

    assert await harness.voices.delete_voice(voice_id, owner=OWNER) == 3
    assert all(uri not in harness.store.blobs for uri in uris)


async def test_delete_is_owner_scoped(harness: Harness) -> None:
    """Keep another caller's voice and recordings."""
    _mine, my_voice = await _submit(harness)
    _theirs, their_voice = await _submit(harness, owner=OTHER, name="theirs.wav")
    await harness.run_queued()
    with pytest.raises(NotFoundError):
        await harness.voices.delete_voice(their_voice, owner=OWNER)
    await harness.voices.delete_voice(my_voice, owner=OWNER)
    remaining = [a for a in harness.uow.db.assets if a.owner == OTHER]
    assert len(remaining) == 2
    assert all(a.asset.stored.location.uri in harness.store.blobs for a in remaining)
    assert harness.uow.db.voices[their_voice].status is VoiceStatus.READY


async def _add_version(harness: Harness, voice_id: VoiceId, *, version: int) -> VoiceVersionId:
    """Store an extra ready version with one recording, as a later re-registration would."""
    latest = harness.uow.db.voices[voice_id].current_version
    assert latest is not None
    version_id = new_voice_version_id()
    harness.uow.db.versions[version_id] = replace(latest, id=version_id, version=version)
    source = write_tone(harness.inputs / f"v{version}.wav", seconds=1.0)
    stored = await harness.store.put(source, key=AssetKey(f"voices/v{version}.wav"), content_type="audio/wav")
    asset = PublishedAsset(
        name="reference",
        role=AssetRole.VOICE_REFERENCE,
        key=AssetKey(f"voices/v{version}.wav"),
        stored=stored,
        content_type="audio/wav",
    )
    harness.uow.db.assets.append(AssetRecord(asset, OWNER, None, version_id))
    return version_id
