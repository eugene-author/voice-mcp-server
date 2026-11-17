"""TranscriptionOptionsResolver: precedence, model availability, substitutions and single/batch parity."""

from pathlib import Path

import pytest

from app.domain.errors import InvalidRequestError, ModelNotInstalledError
from app.domain.jobs import BatchTranscribeCommand, JobType, TranscriptionSpec
from app.domain.models import ModelKind
from app.domain.stt import SttBackendName, SttOptions
from tests.conftest import write_tone
from tests.harness import OWNER, Harness, build_harness
from tests.mocks.stt import MLX_LIKE_CAPABILITIES, MockSttBackend

CT2_SMALL_MEDIUM = {(ModelKind.STT, "ctranslate2", "small"), (ModelKind.STT, "ctranslate2", "medium")}


def _resolve(harness: Harness, **kwargs: object) -> tuple[TranscriptionSpec, list[str]]:
    """Resolve a spec for a fixed input and return it with the substitution descriptions."""
    values: dict[str, object] = {"model": None, "language": None, "preset_name": None, "options": None} | kwargs
    resolved = harness.stt_options.resolve(input_path=Path("/in/a.wav"), **values)  # type: ignore[arg-type]
    return resolved.spec, [item.describe() for item in resolved.substitutions]


def test_caller_overrides_preset_and_preset_overrides_defaults(tmp_path: Path) -> None:
    """Apply caller > preset > server default for model and options."""
    harness = build_harness(tmp_path, installed=CT2_SMALL_MEDIUM)
    default_spec, _ = _resolve(harness)
    preset_spec, _ = _resolve(harness, preset_name="balanced")
    caller_spec, _ = _resolve(
        harness, preset_name="balanced", model="small", options=SttOptions(beam_size=2, word_timestamps=False)
    )
    assert (default_spec.model, default_spec.options.word_timestamps) == ("small", harness.settings.stt.word_timestamps)
    assert (preset_spec.model, preset_spec.options.beam_size, preset_spec.options.word_timestamps) == (
        "medium",
        5,
        True,
    )
    assert (caller_spec.model, caller_spec.options.beam_size, caller_spec.options.word_timestamps) == (
        "small",
        2,
        False,
    )
    harness.executor.shutdown()


def test_missing_models(tmp_path: Path) -> None:
    """Fail for an uninstalled caller model, substitute an uninstalled preset model, reject unknown names."""
    harness = build_harness(tmp_path)
    with pytest.raises(ModelNotInstalledError):
        _resolve(harness, model="medium")
    with pytest.raises(InvalidRequestError):
        _resolve(harness, model="gigantic")
    spec, warnings = _resolve(harness, preset_name="accurate")
    assert spec.model == "small"
    assert warnings == ["model 'large-v3' is not installed; used 'small'"]
    harness.executor.shutdown()


def test_unknown_preset_and_language(tmp_path: Path) -> None:
    """Reject unknown presets and malformed languages; normalize valid language codes."""
    harness = build_harness(tmp_path)
    with pytest.raises(InvalidRequestError, match="unknown preset"):
        _resolve(harness, preset_name="turbo")
    with pytest.raises(InvalidRequestError):
        _resolve(harness, language="english!")
    spec, _ = _resolve(harness, language="pt-BR")
    assert spec.language == "pt"
    harness.executor.shutdown()


def test_mlx_rejects_explicit_unsupported_options_and_substitutes_preset_ones(tmp_path: Path) -> None:
    """Reject caller beam search on MLX but replace a preset's beam size with a warning."""
    harness = build_harness(
        tmp_path,
        stt=MockSttBackend(capabilities=MLX_LIKE_CAPABILITIES, name=SttBackendName.MLX),
        installed={(ModelKind.STT, "mlx", "small")},
    )
    with pytest.raises(InvalidRequestError, match="beam_size"):
        _resolve(harness, options=SttOptions(beam_size=5))
    spec, warnings = _resolve(harness, preset_name="studio")
    assert spec.options.beam_size == 1
    assert any(w.startswith("beam_size 5 not supported by mlx backend") for w in warnings)
    harness.executor.shutdown()


def test_preset_denoise_falls_back_without_denoiser(tmp_path: Path) -> None:
    """Skip preset denoising with a warning when DeepFilterNet is not installed."""
    harness = build_harness(tmp_path, denoiser=False)
    spec, warnings = _resolve(harness, preset_name="studio")
    assert spec.options.denoise is False
    assert "denoise True skipped: DeepFilterNet is not installed; used False" in warnings
    harness.executor.shutdown()


async def test_batch_children_match_single_file_resolution(tmp_path: Path) -> None:
    """Resolve each batch child exactly like a single-file submission with the same preset."""
    harness = build_harness(tmp_path, installed=CT2_SMALL_MEDIUM)
    for name in ("a.wav", "b.wav"):
        write_tone(harness.inputs / name)
    await harness.jobs.submit(BatchTranscribeCommand(directory=harness.inputs, preset="balanced"), owner=OWNER)
    children = [r.job for r in harness.uow.db.jobs.values() if r.job.type is JobType.SPEECH_TO_TEXT]
    assert len(children) == 2
    for child in children:
        single, warnings = _resolve(harness, preset_name="balanced")
        expected = single.to_payload() | {"input_path": child.input_payload["input_path"]}
        assert child.input_payload == expected
        assert [s.describe() for s in child.substitutions] == warnings
    harness.executor.shutdown()
