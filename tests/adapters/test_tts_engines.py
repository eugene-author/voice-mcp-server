"""Torch and MLX Qwen3-TTS adapters with mock qwen_tts and mlx_audio modules."""

import importlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.domain.errors import InferenceError, InvalidRequestError
from app.domain.ids import VoiceVersionId
from app.domain.models import InstalledModel, ModelKind, TtsVariant
from app.domain.tts import CloneReference, PresetSpeaker, SynthesisRequest, TtsBackendName
from tests.mocks import mlx_audio as mock_mlx_audio
from tests.mocks import qwen_tts as mock_qwen

VERSION = VoiceVersionId(uuid.uuid4())


def _checkpoint(tmp_path: Path, variant: TtsVariant, backend: str = "torch") -> InstalledModel:
    """Return a checkpoint directory with a config listing two speakers."""
    directory = tmp_path / variant.value
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"talker_config": {"spk_id": {"vivian": 1, "aiden": 2}}}))
    return InstalledModel(
        name=f"ckpt-{variant.value}", kind=ModelKind.TTS, backend=backend, local_path=directory, variant=variant
    )


def _clone(text: str | None = "reference words") -> SynthesisRequest:
    """Return a clone synthesis request."""
    return SynthesisRequest(
        text="Hello.", language="english", voice=CloneReference(VERSION, Path("/workspace/ref.wav"), text)
    )


def _torch_engine(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, type[mock_qwen.MockQwen3TTSModel]]:
    """Return a torch engine bound to the mock qwen_tts module."""
    model_class = mock_qwen.install(monkeypatch)
    module = importlib.import_module("app.adapters.tts.qwen_torch.engine")
    return module.TorchQwen3TtsEngine(device="cpu", conditioning_cache_size=2), model_class


def test_torch_loads_locally_in_float32_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Load from the local directory without network access."""
    engine, model_class = _torch_engine(monkeypatch)
    engine.load(_checkpoint(tmp_path, TtsVariant.BASE))
    kwargs = model_class.instances[0].load_kwargs
    assert kwargs["device_map"] == "cpu"
    assert kwargs["local_files_only"] is True
    assert str(kwargs["dtype"]) == "torch.float32"
    assert engine.name is TtsBackendName.TORCH


def test_torch_caches_clone_prompt_per_voice_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Build the clone prompt once per voice version and return float32 samples."""
    engine, model_class = _torch_engine(monkeypatch)
    engine.load(_checkpoint(tmp_path, TtsVariant.BASE))
    first = engine.synthesize(_clone())
    engine.synthesize(_clone())
    model = model_class.instances[0]
    assert model.prompts_created == 1
    assert first.sample_rate == 24000
    assert first.samples.dtype == np.float32
    assert model.calls[0][1]["language"] == "english"


def test_torch_xvector_mode_without_transcript(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Request x-vector-only conditioning when the version has no transcript."""
    engine, model_class = _torch_engine(monkeypatch)
    engine.load(_checkpoint(tmp_path, TtsVariant.BASE))
    engine.synthesize(_clone(text=None))
    prompt_kwargs = model_class.instances[0].calls[0][1]["voice_clone_prompt"][0]
    assert prompt_kwargs["x_vector_only_mode"] is True


def test_torch_enforces_checkpoint_variants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Route clones to Base and presets to CustomVoice checkpoints only."""
    engine, _ = _torch_engine(monkeypatch)
    engine.load(_checkpoint(tmp_path, TtsVariant.CUSTOM_VOICE))
    with pytest.raises(InvalidRequestError):
        engine.synthesize(_clone())
    preset = engine.synthesize(SynthesisRequest(text="Hi.", language="english", voice=PresetSpeaker("vivian")))
    assert preset.samples.size == 2400
    engine.load(_checkpoint(tmp_path, TtsVariant.BASE))
    with pytest.raises(InvalidRequestError):
        engine.synthesize(SynthesisRequest(text="Hi.", language="english", voice=PresetSpeaker("vivian")))


def test_torch_preset_speakers_from_config_and_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Read speakers without loading and translate load failures."""
    engine, model_class = _torch_engine(monkeypatch)
    assert engine.preset_speakers(_checkpoint(tmp_path, TtsVariant.CUSTOM_VOICE)) == ("aiden", "vivian")
    assert engine.preset_speakers(_checkpoint(tmp_path, TtsVariant.BASE)) == ()
    with pytest.raises(InferenceError):
        engine.synthesize(_clone())
    model_class.fail_load = OSError("missing")
    with pytest.raises(InferenceError):
        engine.load(_checkpoint(tmp_path, TtsVariant.BASE))


def _mlx_engine(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, mock_mlx_audio.MockMlxAudio]:
    """Return an MLX engine bound to the mock mlx_audio module."""
    recorder = mock_mlx_audio.install(monkeypatch)
    return importlib.import_module("app.adapters.tts.qwen_mlx.engine").MlxQwen3TtsEngine(), recorder


def test_mlx_icl_clone_concatenates_segments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pass reference audio and text, convert segments to one NumPy array and use the model's sample rate."""
    engine, recorder = _mlx_engine(monkeypatch)
    engine.load(_checkpoint(tmp_path, TtsVariant.BASE, backend="mlx"))
    audio = engine.synthesize(_clone())
    kind, kwargs = recorder.models[0].calls[0]
    assert kind == "generate"
    assert kwargs == {
        "text": "Hello.",
        "lang_code": "english",
        "ref_audio": "/workspace/ref.wav",
        "ref_text": "reference words",
    }
    assert recorder.models[0].path == tmp_path / TtsVariant.BASE.value
    assert audio.samples.shape == (150,)
    assert audio.samples.dtype == np.float32
    assert audio.sample_rate == 24000
    assert not engine.capabilities.clone_xvector_only


def test_mlx_rejects_transcript_less_clone_and_supports_presets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refuse x-vector-only references and call generate_custom_voice for presets."""
    engine, recorder = _mlx_engine(monkeypatch)
    engine.load(_checkpoint(tmp_path, TtsVariant.BASE, backend="mlx"))
    with pytest.raises(InvalidRequestError, match="reference_text is required"):
        engine.synthesize(_clone(text=None))
    engine.load(_checkpoint(tmp_path, TtsVariant.CUSTOM_VOICE, backend="mlx"))
    engine.synthesize(SynthesisRequest(text="Hi.", language="english", voice=PresetSpeaker("vivian")))
    assert recorder.models[-1].calls[0] == ("custom", {"text": "Hi.", "speaker": "vivian", "language": "english"})
