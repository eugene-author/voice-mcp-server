"""SQL repositories, unit of work, advisory lock and migrations against real PostgreSQL."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app import main
from app.core.settings import DatabaseSettings
from app.db.locks import PostgresWorkerLock
from app.db.repositories.jobs import MAX_FILES_PER_JOB
from app.db.unit_of_work import SqlUnitOfWorkFactory
from app.domain.assets import (
    INPUT_LOCATION_FIELD,
    AssetKey,
    AssetLocation,
    AssetRole,
    PublishedAsset,
    StorageBackend,
    StoredAsset,
    UploadedInput,
    job_input_key,
    job_output_key,
)
from app.domain.caller import Caller
from app.domain.errors import DuplicateRequestError
from app.domain.ids import JobId, new_claim_token, new_job_id
from app.domain.jobs import (
    ClaimedJob,
    JobCompletion,
    JobCursor,
    JobError,
    JobFilter,
    JobPhase,
    JobStatus,
    JobType,
    NewJob,
)
from app.domain.stt import Segment, Substitution, TranscriptionResult, Word
from app.ports.unit_of_work import UnitOfWork
from tests.harness import Harness, build_harness, host_profile
from tests.integration.conftest import alembic_config, set_claim_token_column

pytestmark = pytest.mark.integration
OWNER = Caller("https://issuer", "alice", "cli")
OTHER = Caller("https://issuer", "bob")
NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


def new_job(
    *, owner: Caller = OWNER, request_id: str | None = None, created_at: datetime = NOW, **kw: object
) -> NewJob:
    """Return a queued speech_to_text job row."""
    return NewJob(
        id=new_job_id(),
        type=kw.pop("type", JobType.SPEECH_TO_TEXT),  # type: ignore[arg-type]
        owner=owner,
        input_payload={"input_path": "/in/a.wav"},
        config_snapshot={"preset": None},
        created_at=created_at,
        max_attempts=2,
        request_id=request_id,
        substitutions=(Substitution("beam_size", 5, 1, "not supported"),),
        **kw,  # type: ignore[arg-type]
    )


async def _claim(uow: UnitOfWork, job: NewJob, *, lease_owner: str = "w") -> ClaimedJob:
    """Claim a job that must be claimable."""
    claimed = await uow.jobs.claim(job.id, lease_owner=lease_owner, now=NOW)
    assert claimed is not None
    return claimed


async def test_add_get_and_ownership(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Persist a job and hide it from other owners."""
    job = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
    async with sql_uow() as uow:
        view = await uow.jobs.get(job.id, OWNER)
        hidden = await uow.jobs.get(job.id, OTHER)
    assert view is not None
    assert view.status is JobStatus.QUEUED
    assert view.warnings == ("beam_size 5 not supported; used 1",)
    assert hidden is None


async def test_request_id_unique_per_owner(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Enforce the partial unique index per owner and translate violations."""
    async with sql_uow() as uow:
        await uow.jobs.add(new_job(request_id="r1"))
        await uow.jobs.add(new_job(owner=OTHER, request_id="r1"))
    with pytest.raises(DuplicateRequestError):
        async with sql_uow() as uow:
            await uow.jobs.add(new_job(request_id="r1"))


async def test_unit_of_work_rolls_back_on_error(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Discard every change of a failed unit of work."""
    job = new_job()

    async def add_then_fail() -> None:
        """Insert a job and raise before the transaction commits."""
        async with sql_uow() as uow:
            await uow.jobs.add(job)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await add_then_fail()
    async with sql_uow() as uow:
        assert await uow.jobs.get(job.id, None) is None


async def test_concurrent_claims_have_one_winner(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Let exactly one of many concurrent claims win."""
    job = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)

    async def claim(worker: int) -> bool:
        """Try to claim the job in its own transaction."""
        async with sql_uow() as uow:
            return await uow.jobs.claim(job.id, lease_owner=f"w{worker}", now=NOW) is not None

    results = await asyncio.gather(*(claim(i) for i in range(8)))
    assert sum(results) == 1


async def test_lifecycle_transitions_are_guarded(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Complete only running jobs and record result, phase and execution."""
    job = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
        assert not await uow.jobs.complete(job.id, JobCompletion(), claim_token=new_claim_token(), now=NOW)
        claimed = await _claim(uow, job)
        token = claimed.claim_token
        assert claimed.attempt_count == 1
        assert await uow.jobs.set_phase(job.id, claim_token=token, phase=JobPhase.INFERENCING, progress=40, now=NOW)
        assert await uow.jobs.heartbeat(job.id, claim_token=token, now=NOW)
        assert not await uow.jobs.heartbeat(job.id, claim_token=new_claim_token(), now=NOW)
    async with sql_uow() as uow:
        running = await uow.jobs.get(job.id, OWNER)
        assert running is not None
        assert running.phase is JobPhase.INFERENCING
        assert running.progress == 40
        completion = JobCompletion(result={"text": "hi"}, warnings=("w1",))
        assert await uow.jobs.complete(job.id, completion, claim_token=token, now=NOW)
        assert not await uow.jobs.fail(job.id, JobError("x", "y"), claim_token=token, now=NOW)
    async with sql_uow() as uow:
        done = await uow.jobs.get(job.id, OWNER)
    assert done is not None
    assert done.status is JobStatus.COMPLETED
    assert done.result == {"text": "hi"}
    assert "w1" in done.warnings
    assert done.progress == 100


async def test_retry_recovery_and_interrupts(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Requeue due retries and interrupt stale leases, failing jobs without attempts left."""
    retry, stale, exhausted = new_job(), new_job(), new_job()
    error = JobError("asset_publish_failed", "x")
    async with sql_uow() as uow:
        claims = {}
        for job in (retry, stale, exhausted):
            await uow.jobs.add(job)
            claims[job.id] = (await _claim(uow, job)).claim_token
        await uow.jobs.schedule_retry(
            retry.id, error, claim_token=claims[retry.id], next_attempt_at=NOW + timedelta(seconds=5)
        )
        await uow.jobs.schedule_retry(exhausted.id, error, claim_token=claims[exhausted.id], next_attempt_at=NOW)
    async with sql_uow() as uow:
        assert await uow.jobs.requeue_due(now=NOW) == 1
        assert await uow.jobs.requeue_due(now=NOW + timedelta(seconds=6)) == 1
        assert set(await uow.jobs.queued_ids(limit=10)) == {retry.id, exhausted.id}
        second_attempt = await uow.jobs.claim(exhausted.id, lease_owner="w", now=NOW)
        assert second_attempt is not None
        assert not second_attempt.has_attempts_left
        interrupted = await uow.jobs.interrupt_stale(heartbeat_before=NOW + timedelta(minutes=5))
        assert set(interrupted) == {stale.id, exhausted.id}
        requeued, failed = await uow.jobs.resolve_interrupted(now=NOW)
    assert requeued == [stale.id]
    assert failed == [exhausted.id]


async def test_pagination_files_and_transcripts(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Page with keyset cursors, expose files only for finished jobs and store transcripts."""
    jobs = [new_job(created_at=NOW + timedelta(seconds=i)) for i in range(3)]
    async with sql_uow() as uow:
        for job in jobs:
            await uow.jobs.add(job)
        claimed = await _claim(uow, jobs[0])
    asset = PublishedAsset(
        name="txt",
        role=AssetRole.OUTPUT,
        key=job_output_key(OWNER, jobs[0].id, "transcript.txt"),
        stored=StoredAsset(AssetLocation(StorageBackend.LOCAL, f"/assets/{jobs[0].id}/t.txt"), "a" * 64, 3),
        content_type="text/plain",
    )
    result = TranscriptionResult(
        segments=(Segment(0, 1, "hi there", words=(Word("hi", 0, 0.5, 0.9), Word("there", 0.5, 1, 0.8))),),
        language="en",
        language_probability=0.9,
        duration_seconds=1,
    )
    async with sql_uow() as uow:
        await uow.assets.add(asset, owner=OWNER, job_id=jobs[0].id, voice_version_id=None, created_at=NOW)
        await uow.assets.add(asset, owner=OWNER, job_id=jobs[0].id, voice_version_id=None, created_at=NOW)
        await uow.transcripts.add(jobs[0].id, OWNER, result, created_at=NOW)
        await uow.jobs.complete(jobs[0].id, JobCompletion(), claim_token=claimed.claim_token, now=NOW)
    async with sql_uow() as uow:
        first = await uow.jobs.page(OWNER, JobFilter(), limit=2, cursor=None)
        assert [v.id for v in first.items] == [jobs[2].id, jobs[1].id]
        assert first.next_cursor
        second = await uow.jobs.page(OWNER, JobFilter(), limit=2, cursor=JobCursor.decode(first.next_cursor))
        completed = await uow.jobs.page(OWNER, JobFilter(status=JobStatus.COMPLETED), limit=5, cursor=None)
    assert [v.id for v in second.items] == [jobs[0].id]
    assert second.next_cursor is None
    assert completed.items[0].files == {"txt": f"/assets/{jobs[0].id}/t.txt"}


async def test_job_files_show_only_public_files_of_finished_jobs(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Project output/manifest files of finished jobs only, capped and in a deterministic order."""
    job = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
        claimed = await _claim(uow, job)
        for index in range(MAX_FILES_PER_JOB + 5):
            name = f"f{index:02d}"
            asset = _asset(job_output_key(OWNER, job.id, f"{name}.txt"), name=name, role=AssetRole.OUTPUT)
            await uow.assets.add(asset, owner=OWNER, job_id=job.id, voice_version_id=None, created_at=NOW)
        original = _asset(job_output_key(OWNER, job.id, "in.wav"), name="original", role=AssetRole.INPUT)
        await uow.assets.add(original, owner=OWNER, job_id=job.id, voice_version_id=None, created_at=NOW)
    async with sql_uow() as uow:
        running = await uow.jobs.get(job.id, OWNER)
        await uow.jobs.complete(job.id, JobCompletion(), claim_token=claimed.claim_token, now=NOW)
    async with sql_uow() as uow:
        finished = await uow.jobs.get(job.id, OWNER)
    assert running is not None
    assert running.files == {}
    assert finished is not None
    assert list(finished.files) == [f"f{index:02d}" for index in range(MAX_FILES_PER_JOB)]
    assert "original" not in finished.files


def _asset(key: AssetKey, *, name: str, role: AssetRole) -> PublishedAsset:
    """Return a published local asset for key."""
    return PublishedAsset(
        name=name,
        role=role,
        key=key,
        stored=StoredAsset(AssetLocation(StorageBackend.LOCAL, f"/assets/{key.value}"), "0" * 64, 1),
        content_type="text/plain",
    )


def _uploaded(job_id: JobId) -> UploadedInput:
    """Return the uploaded original of a job."""
    key = job_input_key(OWNER, job_id, "wav")
    stored = StoredAsset(AssetLocation(StorageBackend.S3, f"s3://bucket/{key.value}"), "0" * 64, 1)
    return UploadedInput(key=key, stored=stored, content_type="audio/wav")


def _uploaded_job() -> tuple[NewJob, str]:
    """Return a queued job whose payload carries an uploaded input, and that input's URI."""
    job = new_job()
    uploaded = _uploaded(job.id)
    payload = {**job.input_payload, INPUT_LOCATION_FIELD: uploaded.to_payload()}
    return replace(job, input_payload=payload), uploaded.stored.location.uri


async def _known(sql_uow: SqlUnitOfWorkFactory, uri: str) -> set[str]:
    """Return the known subset of one URI."""
    async with sql_uow() as uow:
        return await uow.assets.known_uris([uri])


async def test_uploaded_input_is_known_until_its_job_finishes(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Know a queued job's input and forget it once the job fails, as nothing records it."""
    job, uri = _uploaded_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
    assert await _known(sql_uow, uri) == {uri}
    async with sql_uow() as uow:
        claimed = await _claim(uow, job)
    assert await _known(sql_uow, uri) == {uri}
    async with sql_uow() as uow:
        await uow.jobs.fail(job.id, JobError("media_error", "unreadable"), claim_token=claimed.claim_token, now=NOW)
    assert await _known(sql_uow, uri) == set()


async def test_uploaded_input_of_a_retry_wait_job_is_known(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Know the input of a job waiting for its next attempt."""
    job, uri = _uploaded_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
        claimed = await _claim(uow, job)
        await uow.jobs.schedule_retry(
            job.id, JobError("x", "y"), claim_token=claimed.claim_token, next_attempt_at=NOW + timedelta(days=2)
        )
    assert await _known(sql_uow, uri) == {uri}


async def test_known_uris_merges_recorded_assets_and_pending_inputs(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Return recorded and pending URIs together, leaving out every other URI."""
    job, pending = _uploaded_job()
    finished = new_job()
    async with sql_uow() as uow:
        await uow.jobs.add(job)
        await uow.jobs.add(finished)
        await uow.assets.add(
            _uploaded(finished.id).as_original(),
            owner=OWNER,
            job_id=finished.id,
            voice_version_id=None,
            created_at=NOW,
        )
    recorded = _uploaded(finished.id).stored.location.uri
    async with sql_uow() as uow:
        assert await uow.assets.known_uris([pending, recorded, "s3://bucket/orphan"]) == {pending, recorded}


async def test_batch_counts_and_children(sql_uow: SqlUnitOfWorkFactory) -> None:
    """Aggregate child statuses for a batch parent."""
    parent = new_job(type=JobType.TRANSCRIBE_BATCH)
    children = [new_job(parent_job_id=parent.id, created_at=NOW + timedelta(seconds=i)) for i in range(3)]
    async with sql_uow() as uow:
        await uow.jobs.add(parent)
        for child in children:
            await uow.jobs.add(child)
        assert await uow.jobs.claim(parent.id, lease_owner="w", now=NOW) is None
        done = await _claim(uow, children[0])
        await uow.jobs.complete(children[0].id, JobCompletion(), claim_token=done.claim_token, now=NOW)
        bad = await _claim(uow, children[1])
        await uow.jobs.fail(children[1].id, JobError("media_error", "bad"), claim_token=bad.claim_token, now=NOW)
        counts = await uow.jobs.batch_counts(parent.id)
        listed = await uow.jobs.batch_children(parent.id)
        assert await uow.jobs.open_batch_parent_ids() == [parent.id]
    assert counts.to_dict() == {"total": 3, "completed": 1, "failed": 1, "queued": 1}
    assert [c.status for c in listed] == [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.QUEUED]


async def test_advisory_lock_allows_one_worker(engine: AsyncEngine) -> None:
    """Grant the worker lock to one process-level holder at a time."""
    first, second = PostgresWorkerLock(engine), PostgresWorkerLock(engine)
    assert await first.acquire()
    assert not await second.acquire()
    await first.release()
    assert await second.acquire()
    await second.release()


def test_migrations_downgrade_and_upgrade(migrated_database: str, tmp_path: Path) -> None:
    """Apply the migration chain down to base and back up to head."""
    config = alembic_config()
    command.downgrade(config, "base")
    command.upgrade(config, "head")


async def test_cleaned_text_drop_refuses_populated_column(sql_uow: SqlUnitOfWorkFactory, engine: AsyncEngine) -> None:
    """Stop the column-drop migration when cleaned_text holds data, and apply it once the data is gone."""
    config = alembic_config()
    job = new_job()
    result = TranscriptionResult(
        segments=(Segment(0, 1, "hi"),), language="en", language_probability=None, duration_seconds=1
    )
    await asyncio.to_thread(command.downgrade, config, "7c1f0e4b9a21")
    try:
        await set_claim_token_column(engine, "ADD COLUMN claim_token uuid")
        async with sql_uow() as uow:
            await uow.jobs.add(job)
            await uow.transcripts.add(job.id, OWNER, result, created_at=NOW)
        async with engine.begin() as connection:
            await connection.execute(text("UPDATE transcripts SET cleaned_text = 'kept'"))
        with pytest.raises(RuntimeError, match="cleaned_text holds data"):
            await asyncio.to_thread(command.upgrade, config, "head")
        async with engine.begin() as connection:
            await connection.execute(text("UPDATE transcripts SET cleaned_text = NULL"))
    finally:
        await set_claim_token_column(engine, "DROP COLUMN IF EXISTS claim_token")
        await asyncio.to_thread(command.upgrade, config, "head")
    async with engine.connect() as connection:
        columns = await connection.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name = 'transcripts'")
        )
        assert "cleaned_text" not in {row[0] for row in columns}


async def test_diagnose_checks_the_real_database(
    migrated_database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report the database as reachable and release the engine and the inference thread afterwards."""
    harness = build_harness(tmp_path)
    monkeypatch.setattr(main, "build_inference", lambda _settings: _components(harness))
    monkeypatch.setattr(main, "enforce_offline_runtime", lambda: None)
    settings = harness.settings.model_copy(update={"database": DatabaseSettings(url=migrated_database)})  # type: ignore[arg-type]
    async with main.diagnostics_session(settings) as health:
        report = await health.run_diagnostics(include_inference=False)
    assert report.dependencies == {"database": "ok"}
    assert harness.stt.load_count == 0
    with pytest.raises(RuntimeError):
        await harness.executor.run(lambda: None)


def _components(harness: Harness) -> main.InferenceComponents:
    """Return inference components built from the harness."""
    return main.InferenceComponents(
        host=host_profile(),
        stt=harness.stt,
        tts=harness.tts,
        denoiser=None,
        model_host=harness.model_host,
        runtime=harness.local_runtime,
        executor=harness.executor,
        denoise_device=None,
        profile="linux-cpu",
    )
