"""Voice registration, synthesis, listing and deletion through the real services over mocks."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.domain.errors import CapabilityUnavailableError, InvalidRequestError, NotFoundError
from app.domain.ids import VoiceId
from app.domain.jobs import JobStatus, RegisterVoiceCommand, SynthesizeCommand
from app.domain.models import ModelKind
from app.domain.tts import CloneReference, ConditioningMode, PresetSpeaker, TtsOptions
from app.domain.voices import VoiceKind, VoiceStatus
from tests.conftest import write_tone
from tests.harness import DEFAULT_INSTALLED, OTHER, OWNER, Harness, build_harness
from tests.mocks.media import MockMedia
from tests.mocks.stt import MockSttBackend
from tests.mocks.tts import MLX_LIKE, MockTtsEngine

REFERENCE_TEXT = "Hello, it's nice to meet you today."


def _voice_harness(tmp_path: Path, **kwargs: object) -> Harness:
    """Return a harness whose STT hears the reference text and whose media reports an 8 s clip."""
    kwargs.setdefault("stt", MockSttBackend(text=REFERENCE_TEXT))
    kwargs.setdefault("media", MockMedia(duration_seconds=8.0))
    return build_harness(tmp_path, **kwargs)  # type: ignore[arg-type]


async def _register(harness: Harness, *, reference_text: str | None = REFERENCE_TEXT, owner: object = OWNER) -> VoiceId:
    """Register a voice, run the job and return the voice ID."""
    path = write_tone(harness.inputs / "reference.wav", seconds=8.0, sample_rate=24000)
    command = RegisterVoiceCommand(
        name="Narrator", reference_audio_path=path, consent_confirmed=True, reference_text=reference_text, language="en"
    )
    submission = await harness.jobs.submit(command, owner=owner)  # type: ignore[arg-type]
    await harness.run_queued()
    assert submission.voice_id is not None
    return submission.voice_id


@pytest.fixture
def voices(tmp_path: Path) -> Iterator[Harness]:
    """Return a voice-ready harness."""
    harness = _voice_harness(tmp_path)
    yield harness
    harness.executor.shutdown()


async def test_registration_publishes_reference_and_marks_ready(voices: Harness) -> None:
    """Make the voice ready with a published reference and portable version data only."""
    voice_id = await _register(voices)
    record = voices.uow.db.voices[voice_id]
    assert record.status is VoiceStatus.READY
    version = record.current_version
    assert version is not None
    assert version.reference_uri is not None
    assert version.conditioning_mode is ConditioningMode.ICL
    roles = {a.asset.role.value for a in voices.uow.db.assets}
    assert roles == {"voice_reference", "input"}


async def test_consent_is_required(voices: Harness) -> None:
    """Reject registrations without affirmative consent."""
    path = write_tone(voices.inputs / "r.wav")
    with pytest.raises(InvalidRequestError, match="consent"):
        await voices.jobs.submit(
            RegisterVoiceCommand(name="x", reference_audio_path=path, consent_confirmed=False), owner=OWNER
        )


async def test_transcript_mismatch_fails_registration(tmp_path: Path) -> None:
    """Fail the job and the voice when the transcript does not match the recording."""
    harness = _voice_harness(tmp_path, stt=MockSttBackend(text="something entirely unrelated here"))
    voice_id = await _register(harness)
    assert harness.uow.db.voices[voice_id].status is VoiceStatus.FAILED
    job = next(iter(harness.uow.db.jobs.values()))
    assert job.status is JobStatus.FAILED
    assert job.error is not None
    assert "does not match" in job.error.message
    harness.executor.shutdown()


async def test_reference_duration_bounds(tmp_path: Path) -> None:
    """Reject references shorter than the minimum."""
    harness = _voice_harness(tmp_path, media=MockMedia(duration_seconds=1.0))
    voice_id = await _register(harness)
    job = next(iter(harness.uow.db.jobs.values()))
    assert harness.uow.db.voices[voice_id].status is VoiceStatus.FAILED
    assert job.error is not None
    assert job.error.code == "invalid_request"
    harness.executor.shutdown()


async def test_xvector_registration_only_where_supported(tmp_path: Path) -> None:
    """Allow transcript-less registration on torch-like runtimes and reject it on mlx-like ones."""
    torch_like = _voice_harness(tmp_path / "a")
    voice_id = await _register(torch_like, reference_text=None)
    version = torch_like.uow.db.voices[voice_id].current_version
    assert version is not None
    assert version.conditioning_mode is ConditioningMode.XVECTOR
    torch_like.executor.shutdown()
    mlx_like = _voice_harness(tmp_path / "b", tts=MockTtsEngine(capabilities=MLX_LIKE))
    with pytest.raises(InvalidRequestError, match="reference_text is required"):
        await _register(mlx_like, reference_text=None)
    mlx_like.executor.shutdown()


async def test_synthesis_pins_voice_version_and_publishes_audio(voices: Harness) -> None:
    """Synthesize every chunk with the pinned version and publish the encoded output."""
    voice_id = await _register(voices)
    text = "First sentence. " * 30
    submission = await voices.jobs.submit(
        SynthesizeCommand(text=text, voice_id=voice_id, options=TtsOptions(output_format="mp3", max_chunk_chars=100)),
        owner=OWNER,
    )
    await voices.run_queued()
    view = await voices.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.COMPLETED
    assert set(view.files) == {"mp3"}
    assert voices.tts is not None
    synth_requests = [r for r in voices.tts.requests if r.text != "This is a short check of the new voice."]
    assert len(synth_requests) > 1
    assert all(isinstance(r.voice, CloneReference) for r in synth_requests)
    pinned = voices.uow.db.voices[voice_id].current_version
    assert pinned is not None
    assert {r.voice.voice_version_id for r in synth_requests} == {pinned.id}  # type: ignore[union-attr]
    assert voices.media.encoded[-1][1] == "mp3"


async def test_synthesis_with_preset_speaker(voices: Harness) -> None:
    """Use the CustomVoice checkpoint for preset speakers."""
    submission = await voices.jobs.submit(SynthesizeCommand(text="Hello there.", voice_id="preset:vivian"), owner=OWNER)
    await voices.run_queued()
    assert (await voices.queries.get(submission.job_id, owner=OWNER)).status is JobStatus.COMPLETED
    assert voices.tts is not None
    assert voices.tts.requests[-1].voice == PresetSpeaker("vivian")


async def test_list_voices_includes_presets_and_owned_clones(voices: Harness) -> None:
    """List preset speakers plus only the caller's cloned voices."""
    mine = await _register(voices)
    await _register(voices, owner=OTHER)
    listed = await voices.voices.list_voices(owner=OWNER, language=None)
    assert [v.voice_id for v in listed if v.kind is VoiceKind.PRESET] == ["preset:vivian", "preset:ryan"]
    assert [v.voice_id for v in listed if v.kind is VoiceKind.CLONED] == [mine]


async def test_list_voices_without_custom_voice_checkpoint(tmp_path: Path) -> None:
    """Omit presets when the CustomVoice checkpoint is not installed."""
    installed = {item for item in DEFAULT_INSTALLED if "CustomVoice" not in item[2]}
    harness = _voice_harness(tmp_path, installed=installed)
    assert await harness.voices.list_voices(owner=OWNER, language=None) == []
    harness.executor.shutdown()


async def test_other_owners_voice_is_not_found(voices: Harness) -> None:
    """Hide voices of other callers."""
    voice_id = await _register(voices, owner=OTHER)
    with pytest.raises(NotFoundError):
        await voices.jobs.submit(SynthesizeCommand(text="hi", voice_id=voice_id), owner=OWNER)


async def test_xvector_voice_unusable_on_mlx_like_runtime(tmp_path: Path) -> None:
    """Return invalid_request with a re-register hint for x-vector voices on ICL-only runtimes."""
    harness = _voice_harness(tmp_path)
    voice_id = await _register(harness, reference_text=None)
    mlx_like = _voice_harness(tmp_path / "mlx", tts=MockTtsEngine(capabilities=MLX_LIKE))
    mlx_like.uow.db = harness.uow.db
    with pytest.raises(InvalidRequestError, match="re-register"):
        await mlx_like.jobs.submit(SynthesizeCommand(text="hi", voice_id=voice_id), owner=OWNER)
    harness.executor.shutdown()
    mlx_like.executor.shutdown()


async def test_text_limits(voices: Harness) -> None:
    """Reject empty and over-long text and unsupported languages."""
    for command in (
        SynthesizeCommand(text="   ", voice_id="preset:vivian"),
        SynthesizeCommand(text="x" * 20001, voice_id="preset:vivian"),
        SynthesizeCommand(text="hi", voice_id="preset:vivian", language="xx"),
    ):
        with pytest.raises(InvalidRequestError):
            await voices.jobs.submit(command, owner=OWNER)


async def test_silent_chunk_fails_after_retries(tmp_path: Path) -> None:
    """Detect skipped or silent chunks instead of publishing broken audio."""
    harness = _voice_harness(tmp_path, tts=MockTtsEngine(silent=True))
    submission = await harness.jobs.submit(SynthesizeCommand(text="Hello.", voice_id="preset:vivian"), owner=OWNER)
    await harness.run_queued()
    view = await harness.queries.get(submission.job_id, owner=OWNER)
    assert view.status is JobStatus.RETRY_WAIT
    assert view.error is not None
    assert view.error.code == "inference_error"
    harness.executor.shutdown()


async def test_delete_voice_purges_reference_and_preserved_original(voices: Harness) -> None:
    """Delete both stored recordings of a registration, keep the caller's file, and hide the voice."""
    voice_id = await _register(voices)
    caller_file = voices.inputs / "reference.wav"
    recordings = {a.asset.role.value: a.asset.stored.location.uri for a in voices.uow.db.assets}
    assert set(recordings) == {"voice_reference", "input"}
    assert all(a.voice_version_id is not None for a in voices.uow.db.assets)
    assert all(uri in voices.store.blobs for uri in recordings.values())

    removed = await voices.voices.delete_voice(voice_id, owner=OWNER)

    assert removed == 2
    assert all(uri not in voices.store.blobs for uri in recordings.values())
    assert voices.uow.db.assets == []
    assert caller_file.is_file()
    version = voices.uow.db.versions[voices.uow.db.voices[voice_id].current_version.id]  # type: ignore[union-attr]
    assert (version.reference_uri, version.reference_backend, version.reference_sha256) == (None, None, None)
    listed = await voices.voices.list_voices(owner=OWNER, language=None)
    assert [v for v in listed if v.kind is VoiceKind.CLONED] == []
    with pytest.raises(NotFoundError):
        await voices.jobs.submit(SynthesizeCommand(text="hi", voice_id=voice_id), owner=OWNER)


async def test_tts_unavailable_without_runtime(tmp_path: Path) -> None:
    """Return capability_unavailable for TTS commands without a runtime."""
    harness = build_harness(tmp_path, tts=None, installed={(ModelKind.STT, "ctranslate2", "small")})
    with pytest.raises(CapabilityUnavailableError):
        await harness.jobs.submit(SynthesizeCommand(text="hi", voice_id="preset:vivian"), owner=OWNER)
    harness.executor.shutdown()
