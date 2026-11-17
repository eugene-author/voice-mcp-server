"""Voice recording cleanup, registration job/voice consistency and the input-link backfill on real PostgreSQL."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.unit_of_work import SqlUnitOfWorkFactory
from app.domain.assets import (
    AssetKey,
    AssetRole,
    PublishedAsset,
    job_input_key,
    job_output_key,
    voice_reference_key,
)
from app.domain.caller import Caller
from app.domain.errors import InvalidRequestError, NotFoundError
from app.domain.ids import VoiceId, VoiceVersionId, new_claim_token, new_job_id, new_voice_id, new_voice_version_id
from app.domain.jobs import JobCompletion, JobStatus, JobType, NewJob
from app.domain.tts import ConditioningMode
from app.domain.voices import ConsentRecord, NewVoice, VoiceStatus
from app.services.batch_service import BatchService
from app.services.job_workspace import JobWorkspace
from app.services.publishing import AssetPublisher
from app.services.voice_service import VoiceService
from app.workers.job_transitions import JobTransitions
from app.workers.recovery import JobRecovery
from tests.harness import Harness, build_harness
from tests.integration.conftest import alembic_config, set_claim_token_column
from tests.mocks.clock import MockClock
from tests.mocks.queue import MockJobQueue, MockJobWaiters
from tests.mocks.storage import MockAssetStore

pytestmark = pytest.mark.integration
OWNER = Caller("https://issuer", "alice", "cli")
OTHER = Caller("https://issuer", "bob")
NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
PRE_FIX_REVISION = "2d05911d2db1"


@dataclass
class Registration:
    """A submitted register_voice job and its pending voice."""

    job: NewJob
    voice_id: VoiceId
    version_id: VoiceVersionId
    owner: Caller


@dataclass
class World:
    """Real SQL job components over a mock store and clock."""

    uow: SqlUnitOfWorkFactory
    store: MockAssetStore
    clock: MockClock
    transitions: JobTransitions
    recovery: JobRecovery
    voices: VoiceService
    tmp: Path


@pytest.fixture
async def world(sql_uow: SqlUnitOfWorkFactory, tmp_path: Path) -> AsyncIterator[World]:
    """Return transitions, recovery and voice services wired to PostgreSQL."""
    harness: Harness = build_harness(tmp_path / "harness")
    store, clock = MockAssetStore(), MockClock(NOW)
    batches = BatchService(
        uow=sql_uow,
        publisher=AssetPublisher(store),
        workspace=JobWorkspace(tmp_path / "workspace"),
        waiters=MockJobWaiters(),
        clock=clock,
    )
    yield World(
        uow=sql_uow,
        store=store,
        clock=clock,
        transitions=JobTransitions(uow=sql_uow, clock=clock, retry_backoff_seconds=(1.0,), save_events=True),
        recovery=JobRecovery(
            uow=sql_uow,
            queue=MockJobQueue(),
            batches=batches,
            clock=clock,
            lease_timeout_seconds=60,
            poll_interval_seconds=60,
        ),
        voices=VoiceService(
            uow=sql_uow,
            store=store,
            runtime=harness.runtime,
            preset_checkpoint=None,
            clock=clock,
        ),
        tmp=tmp_path,
    )
    harness.executor.shutdown()


async def _submit(world: World, *, owner: Caller = OWNER, max_attempts: int = 2) -> Registration:
    """Insert a pending voice and its queued register_voice job."""
    voice_id, version_id = new_voice_id(), new_voice_version_id()
    job = NewJob(
        id=new_job_id(),
        type=JobType.REGISTER_VOICE,
        owner=owner,
        input_payload={"voice_id": voice_id, "voice_version_id": str(version_id), "reference_audio_path": "/in/r.wav"},
        config_snapshot={},
        created_at=NOW,
        max_attempts=max_attempts,
    )
    async with world.uow() as uow:
        await uow.voices.add(
            NewVoice(
                voice_id=voice_id,
                voice_version_id=version_id,
                owner=owner,
                name="Narrator",
                language="en",
                reference_text="hello",
                conditioning_mode=ConditioningMode.ICL,
                consent=ConsentRecord(confirmed=True, confirmed_at=NOW, source="test"),
                created_at=NOW,
            )
        )
        await uow.jobs.add(job)
    return Registration(job=job, voice_id=voice_id, version_id=version_id, owner=owner)


async def _publish(world: World, key: AssetKey, *, name: str, role: AssetRole) -> PublishedAsset:
    """Store a small file under key and return its published asset."""
    source = world.tmp / f"{name}-{key.value.replace('/', '_')}"
    source.write_bytes(b"RIFF" + name.encode())
    stored = await world.store.put(source, key=key, content_type="audio/wav")
    return PublishedAsset(name=name, role=role, key=key, stored=stored, content_type="audio/wav")


async def _run_to_completion(world: World, registration: Registration) -> JobCompletion:
    """Claim the job, publish both recordings like the registration service, and complete it."""
    claimed = await world.transitions.claim(registration.job.id, lease_owner="worker")
    assert claimed is not None
    completion = await _completion(world, registration)
    assert await world.transitions.mark_completed(claimed, completion) is JobStatus.COMPLETED
    return completion


async def _completion(world: World, registration: Registration) -> JobCompletion:
    """Return a registration completion with the processed reference and the preserved original."""
    owner, voice_id, job_id = registration.owner, registration.voice_id, registration.job.id
    reference = await _publish(
        world,
        voice_reference_key(owner, voice_id, 1, "wav", claim_token=new_claim_token()),
        name="reference",
        role=AssetRole.VOICE_REFERENCE,
    )
    original = await _publish(world, job_input_key(owner, job_id, "wav"), name="original", role=AssetRole.INPUT)
    return JobCompletion(assets=(reference, original), voice_version_id=registration.version_id)


async def _states(world: World, registration: Registration) -> tuple[JobStatus, VoiceStatus]:
    """Return the stored job and voice status."""
    async with world.uow() as uow:
        job = await uow.jobs.get(registration.job.id, None)
        voice = await uow.voices.get(registration.voice_id, None, include_deleted=True)
    assert job is not None
    assert voice is not None
    return job.status, voice.status


async def _interrupt(world: World, registration: Registration) -> None:
    """Claim as a worker that dies, then let recovery handle the expired lease."""
    await world.transitions.claim(registration.job.id, lease_owner="dead-worker")
    world.clock.advance(seconds=600)
    await world.recovery.recover()


async def test_completion_links_both_recordings_and_delete_purges_them(world: World) -> None:
    """Attach reference and original to the version, then delete both rows and objects on voice deletion."""
    registration = await _submit(world)
    completion = await _run_to_completion(world, registration)
    async with world.uow() as uow:
        locations = await uow.assets.locations_for_voice(registration.voice_id)
    assert {loc.uri for loc in locations} == {a.stored.location.uri for a in completion.assets}

    assert await world.voices.delete_voice(registration.voice_id, owner=OWNER) == 2

    assert all(a.stored.location.uri not in world.store.blobs for a in completion.assets)
    async with world.uow() as uow:
        assert await uow.assets.locations_for_voice(registration.voice_id) == []
        assert await uow.voices.get(registration.voice_id, OWNER) is None
        version = await uow.voices.get_version(registration.version_id)
    assert version is not None
    assert (version.reference_uri, version.reference_backend, version.reference_sha256) == (None, None, None)


async def test_delete_leaves_other_voices_and_callers_alone(world: World) -> None:
    """Purge only the selected voice's recordings, not another owner's or unrelated job assets."""
    mine, theirs = await _submit(world), await _submit(world, owner=OTHER)
    await _run_to_completion(world, mine)
    their_completion = await _run_to_completion(world, theirs)
    unrelated = await _publish(
        world, job_output_key(OWNER, mine.job.id, "transcript.txt"), name="txt", role=AssetRole.OUTPUT
    )
    async with world.uow() as uow:
        await uow.assets.add(unrelated, owner=OWNER, job_id=mine.job.id, voice_version_id=None, created_at=NOW)

    with pytest.raises(NotFoundError):
        await world.voices.delete_voice(theirs.voice_id, owner=OWNER)
    await world.voices.delete_voice(mine.voice_id, owner=OWNER)

    assert unrelated.stored.location.uri in world.store.blobs
    assert all(a.stored.location.uri in world.store.blobs for a in their_completion.assets)
    async with world.uow() as uow:
        assert len(await uow.assets.locations_for_voice(theirs.voice_id)) == 2
        assert await uow.assets.known_uris([unrelated.stored.location.uri]) == {unrelated.stored.location.uri}


async def test_delete_retries_after_partial_storage_failure(world: World) -> None:
    """Keep rows when storage fails midway so a second delete finishes the purge."""
    registration = await _submit(world)
    completion = await _run_to_completion(world, registration)
    world.store.fail_deletes_after = 1
    with pytest.raises(Exception, match="simulated delete failure"):
        await world.voices.delete_voice(registration.voice_id, owner=OWNER)
    async with world.uow() as uow:
        assert len(await uow.assets.locations_for_voice(registration.voice_id)) == 2
        assert await uow.voices.get(registration.voice_id, OWNER) is None

    world.store.fail_deletes_after = None
    assert await world.voices.delete_voice(registration.voice_id, owner=OWNER) == 2
    assert all(a.stored.location.uri not in world.store.blobs for a in completion.assets)


async def test_pending_voice_cannot_be_deleted(world: World) -> None:
    """Refuse to delete a voice whose registration has not finished."""
    registration = await _submit(world)
    with pytest.raises(InvalidRequestError):
        await world.voices.delete_voice(registration.voice_id, owner=OWNER)
    assert await _states(world, registration) == (JobStatus.QUEUED, VoiceStatus.PENDING)


async def test_exhausted_interrupted_registration_fails_voice(world: World) -> None:
    """Fail the voice in the same recovery transaction that fails the exhausted job."""
    registration = await _submit(world)
    await _interrupt(world, registration)
    assert await _states(world, registration) == (JobStatus.QUEUED, VoiceStatus.PENDING)
    await _interrupt(world, registration)
    assert await _states(world, registration) == (JobStatus.FAILED, VoiceStatus.FAILED)


async def test_manual_retry_then_success_makes_voice_ready(world: World) -> None:
    """Reset the failed voice to pending on retry; success then makes it ready with its reference."""
    registration = await _submit(world)
    claimed = await world.transitions.claim(registration.job.id, lease_owner="worker")
    assert claimed is not None
    await world.transitions.mark_failed(claimed, InvalidRequestError("bad reference"))
    assert await _states(world, registration) == (JobStatus.FAILED, VoiceStatus.FAILED)

    assert await world.transitions.retry(registration.job.id)
    assert await _states(world, registration) == (JobStatus.QUEUED, VoiceStatus.PENDING)
    await _run_to_completion(world, registration)
    assert await _states(world, registration) == (JobStatus.COMPLETED, VoiceStatus.READY)


async def test_cancel_fails_voice_and_deleted_voice_is_not_revived(world: World) -> None:
    """Fail the voice of a cancelled registration, and refuse a retry once the voice is deleted."""
    registration = await _submit(world)
    assert await world.transitions.cancel(registration.job.id)
    assert await _states(world, registration) == (JobStatus.CANCELLED, VoiceStatus.FAILED)

    second = await _submit(world)
    claimed = await world.transitions.claim(second.job.id, lease_owner="worker")
    assert claimed is not None
    await world.transitions.mark_failed(claimed, InvalidRequestError("bad reference"))
    await world.voices.delete_voice(second.voice_id, owner=OWNER)
    assert not await world.transitions.retry(second.job.id)
    assert await _states(world, second) == (JobStatus.FAILED, VoiceStatus.DELETED)


async def test_late_completion_of_deleted_voice_fails_job_without_records(world: World) -> None:
    """Roll back a completion whose voice was deleted meanwhile and leave the references cleared."""
    registration = await _submit(world)
    claimed = await world.transitions.claim(registration.job.id, lease_owner="worker")
    assert claimed is not None
    async with world.uow() as uow:
        await uow.voices.mark_failed(registration.voice_id, now=NOW)
        await uow.voices.mark_deleted(registration.voice_id, now=NOW)
    completion = await _completion(world, registration)

    assert await world.transitions.mark_completed(claimed, completion) is JobStatus.FAILED

    assert await _states(world, registration) == (JobStatus.FAILED, VoiceStatus.DELETED)
    async with world.uow() as uow:
        assert await uow.assets.known_uris([a.stored.location.uri for a in completion.assets]) == set()
        version = await uow.voices.get_version(registration.version_id)
    assert version is not None
    assert version.reference_uri is None


async def test_backfill_links_legacy_registration_originals(world: World, engine: AsyncEngine) -> None:
    """Upgrade a pre-fix database: link verified originals only, then deletion purges them."""
    config = alembic_config()
    await asyncio.to_thread(command.downgrade, config, PRE_FIX_REVISION)
    try:
        await set_claim_token_column(engine, "ADD COLUMN claim_token uuid")
        legacy = await _submit(world)
        legacy_completion = await _completion(world, legacy)
        malformed = await _submit(world)
        malformed_original = await _publish(
            world, job_input_key(OWNER, malformed.job.id, "wav"), name="original", role=AssetRole.INPUT
        )
        stt_job = NewJob(
            id=new_job_id(),
            type=JobType.SPEECH_TO_TEXT,
            owner=OWNER,
            input_payload={"input_path": "/in/a.wav"},
            config_snapshot={},
            created_at=NOW,
            max_attempts=2,
        )
        stt_original = await _publish(
            world, job_input_key(OWNER, stt_job.id, "wav"), name="original", role=AssetRole.INPUT
        )
        async with world.uow() as uow:
            await uow.jobs.add(stt_job)
            reference, original = legacy_completion.assets
            await uow.assets.add(
                reference, owner=OWNER, job_id=legacy.job.id, voice_version_id=legacy.version_id, created_at=NOW
            )
            await uow.assets.add(original, owner=OWNER, job_id=legacy.job.id, voice_version_id=None, created_at=NOW)
            await uow.assets.add(
                malformed_original, owner=OWNER, job_id=malformed.job.id, voice_version_id=None, created_at=NOW
            )
            await uow.assets.add(stt_original, owner=OWNER, job_id=stt_job.id, voice_version_id=None, created_at=NOW)
            await uow.voices.mark_ready(
                legacy.version_id, reference=reference.stored.location, sha256="0" * 64, now=NOW
            )
        async with engine.begin() as connection:
            await connection.execute(
                text("UPDATE jobs SET input_payload = input_payload || CAST(:patch AS jsonb) WHERE id = :id"),
                {"patch": '{"voice_version_id": "not-a-uuid"}', "id": malformed.job.id},
            )
    finally:
        await set_claim_token_column(engine, "DROP COLUMN IF EXISTS claim_token")
        await asyncio.to_thread(command.upgrade, config, "head")

    async with world.uow() as uow:
        linked = {loc.uri for loc in await uow.assets.locations_for_voice(legacy.voice_id)}
        assert await uow.assets.locations_for_voice(malformed.voice_id) == []
    assert linked == {a.stored.location.uri for a in legacy_completion.assets}

    assert await world.voices.delete_voice(legacy.voice_id, owner=OWNER) == 2
    assert all(a.stored.location.uri not in world.store.blobs for a in legacy_completion.assets)
    assert stt_original.stored.location.uri in world.store.blobs
    async with world.uow() as uow:
        assert await uow.assets.known_uris(
            [stt_original.stored.location.uri, malformed_original.stored.location.uri]
        ) == {
            stt_original.stored.location.uri,
            malformed_original.stored.location.uri,
        }


async def test_other_job_types_do_not_touch_voices(world: World) -> None:
    """Cancel a transcription job without changing any voice."""
    registration = await _submit(world)
    stt_job = NewJob(
        id=new_job_id(),
        type=JobType.SPEECH_TO_TEXT,
        owner=OWNER,
        input_payload={"voice_id": registration.voice_id},
        config_snapshot={},
        created_at=NOW + timedelta(seconds=1),
        max_attempts=2,
    )
    async with world.uow() as uow:
        await uow.jobs.add(stt_job)
    assert await world.transitions.cancel(stt_job.id)
    assert await _states(world, registration) == (JobStatus.QUEUED, VoiceStatus.PENDING)


async def test_stale_claim_cannot_finish_a_registration(world: World) -> None:
    """Change no voice, asset or event when a claim that lost the job completes or fails it, then let the new one."""
    registration = await _submit(world)
    stale = await world.transitions.claim(registration.job.id, lease_owner="same:1")
    assert stale is not None
    world.clock.advance(seconds=61)
    await world.recovery.recover()
    current = await world.transitions.claim(registration.job.id, lease_owner="same:1")
    assert current is not None
    late = await _completion(world, registration)
    assert await world.transitions.mark_completed(stale, late) is None
    assert await world.transitions.mark_failed(stale, InvalidRequestError("stale")) is None
    assert await _states(world, registration) == (JobStatus.RUNNING, VoiceStatus.PENDING)
    async with world.uow() as uow:
        assert await uow.assets.known_uris([asset.stored.location.uri for asset in late.assets]) == set()
    accepted = await _completion(world, registration)
    assert await world.transitions.mark_completed(current, accepted) is JobStatus.COMPLETED
    assert await _states(world, registration) == (JobStatus.COMPLETED, VoiceStatus.READY)
