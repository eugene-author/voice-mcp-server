"""Local and registry runtime catalogs, and the api role validating requests from a worker's published snapshot."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.domain.errors import CapabilityUnavailableError, InvalidRequestError
from app.domain.jobs import JobStatus, RegisterVoiceCommand, SynthesizeCommand, TranscribeCommand
from app.domain.models import ModelKind
from app.domain.stt import SttOptions, SttRequest, VadSettings
from app.domain.voices import VoiceKind
from app.domain.workers import CapabilitySnapshot
from app.services.runtime_catalog import NO_WORKER_MESSAGE, RegistryRuntimeCatalog
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness, publish_worker


@pytest.fixture
def worker(tmp_path: Path) -> Iterator[Harness]:
    """Return a harness standing for the GPU worker: models installed, TTS and a denoiser present."""
    harness = build_harness(tmp_path / "worker", denoiser=True)
    yield harness
    harness.executor.shutdown()


@pytest.fixture
def api(tmp_path: Path) -> Iterator[Harness]:
    """Return a harness standing for the api role: no models, no TTS runtime, services on the registry catalog."""
    harness = build_harness(tmp_path / "api", tts=None, installed=set(), registry=True)
    yield harness
    harness.executor.shutdown()


async def _registry_for(worker: Harness) -> RegistryRuntimeCatalog:
    """Publish the worker's snapshot and return a refreshed registry catalog over the same database."""
    await publish_worker(worker.uow, worker.local_runtime.snapshot(), at=worker.clock.now())
    registry = RegistryRuntimeCatalog(uow=worker.uow, refresh_seconds=15)
    await registry.refresh()
    return registry


async def _publish_to(api: Harness, worker: Harness) -> None:
    """Publish the worker's snapshot into the api's database and refresh the api's catalog."""
    await publish_worker(api.uow, worker.local_runtime.snapshot(), at=api.clock.now())
    assert isinstance(api.runtime, RegistryRuntimeCatalog)
    await api.runtime.refresh()


def test_snapshot_round_trips_through_json(worker: Harness) -> None:
    """Rebuild an identical snapshot from its JSON form."""
    snapshot = worker.local_runtime.snapshot()
    assert CapabilitySnapshot.from_dict(snapshot.to_dict()) == snapshot
    assert snapshot.preset_speakers == {"Qwen3-TTS-12Hz-0.6B-CustomVoice": ("vivian", "ryan")}


async def test_local_and_registry_catalogs_answer_alike(worker: Harness) -> None:
    """Give the same answers from the live components and from the published snapshot."""
    local, registry = worker.local_runtime, await _registry_for(worker)
    assert registry.snapshot() == local.snapshot()
    assert registry.stt_runtime() == local.stt_runtime()
    assert registry.stt_capabilities() == local.stt_capabilities()
    assert registry.tts_backend() == local.tts_backend()
    assert registry.tts_capabilities() == local.tts_capabilities()
    assert registry.denoise_available() is local.denoise_available() is True
    assert registry.installed_stt_names() == local.installed_stt_names() == {"small"}
    assert registry.loaded_models() == local.loaded_models()
    for kind, backend, name in [
        (ModelKind.TTS, "torch", "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        (ModelKind.STT, "ctranslate2", "medium"),
        (ModelKind.DENOISE, "deepfilternet", "deepfilternet3"),
    ]:
        assert registry.has_model(kind, backend, name) is local.has_model(kind, backend, name)
    checkpoint = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert registry.preset_speakers(checkpoint) == local.preset_speakers(checkpoint)


async def test_without_a_registered_worker_submissions_fail(api: Harness) -> None:
    """Fail submissions with capability_unavailable and the register-only hint until a worker has registered."""
    assert isinstance(api.runtime, RegistryRuntimeCatalog)
    await api.runtime.refresh()
    with pytest.raises(CapabilityUnavailableError, match="worker --register-only") as caught:
        await api.jobs.submit(TranscribeCommand(input_path=write_tone(api.inputs / "a.wav")), owner=OWNER)
    assert caught.value.message == NO_WORKER_MESSAGE
    assert api.uow.db.jobs == {}


async def test_api_validates_from_the_snapshot_without_models(api: Harness, worker: Harness) -> None:
    """Plan transcription, preset synthesis and registration on a host with no models or TTS runtime."""
    await _publish_to(api, worker)
    stt = await api.jobs.submit(
        TranscribeCommand(input_path=write_tone(api.inputs / "a.wav"), options=SttOptions(denoise=True)),
        owner=OWNER,
    )
    tts = await api.jobs.submit(SynthesizeCommand(text="Hello.", voice_id="preset:vivian"), owner=OWNER)
    voice = await api.jobs.submit(
        RegisterVoiceCommand(
            name="me",
            reference_audio_path=write_tone(api.inputs / "ref.wav", seconds=8),
            reference_text="hello there",
            consent_confirmed=True,
        ),
        owner=OWNER,
    )
    for submission in (stt, tts, voice):
        assert (await api.queries.get(submission.job_id, owner=OWNER)).status is JobStatus.QUEUED
    snapshot = api.uow.db.jobs[tts.job_id].job.config_snapshot
    assert (snapshot["profile_stt_backend"], snapshot["tts_backend"]) == ("ctranslate2", "torch")


async def test_api_rejects_what_the_worker_cannot_run(api: Harness, tmp_path: Path) -> None:
    """Reject TTS when the worker has no TTS runtime, and unknown STT models, from the snapshot alone."""
    stt_only = build_harness(tmp_path / "stt-only", tts=None)
    try:
        await _publish_to(api, stt_only)
    finally:
        stt_only.executor.shutdown()
    with pytest.raises(CapabilityUnavailableError, match="'tts' extra"):
        await api.jobs.submit(SynthesizeCommand(text="Hello.", voice_id="preset:vivian"), owner=OWNER)
    with pytest.raises(InvalidRequestError, match="model must be one of"):
        await api.jobs.submit(TranscribeCommand(input_path=write_tone(api.inputs / "a.wav"), model="huge"), owner=OWNER)


async def test_api_lists_preset_voices_from_the_snapshot(api: Harness, worker: Harness) -> None:
    """List the worker's preset speakers on a host that cannot read any checkpoint."""
    await _publish_to(api, worker)
    voices = await api.voices.list_voices(owner=OWNER, language=None)
    assert [(v.voice_id, v.kind) for v in voices] == [
        ("preset:vivian", VoiceKind.PRESET),
        ("preset:ryan", VoiceKind.PRESET),
    ]


async def test_list_models_reports_resident_models_live(worker: Harness) -> None:
    """Report a model loaded after the model service was built (it used to keep the startup value forever)."""
    models = worker.tool_services().models
    assert not any(status.loaded for status in models.list_models(ModelKind.STT))
    await worker.model_host.transcribe(
        worker.model_host.stt_model("small"),
        SttRequest(
            audio_path=write_tone(worker.inputs / "a.wav"),
            language="en",
            beam_size=1,
            temperature=0.0,
            word_timestamps=False,
            vad=VadSettings(),
        ),
    )
    loaded = [status.name for status in models.list_models(ModelKind.STT) if status.loaded]
    assert loaded == ["small"]
