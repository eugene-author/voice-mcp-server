"""Qwen3-TTS through the official qwen-tts package on PyTorch (CUDA or CPU)."""

import gc
import json
import logging
from typing import Any

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from app.adapters.tts.qwen_torch.conditioning import VoiceConditioningCache
from app.domain.errors import InferenceError, InvalidRequestError
from app.domain.models import InstalledModel, TtsVariant
from app.domain.tts import (
    CloneReference,
    PresetSpeaker,
    SynthesisRequest,
    SynthesizedAudio,
    TtsBackendName,
    TtsCapabilities,
)

logger = logging.getLogger(__name__)

CAPABILITIES = TtsCapabilities(
    clone_icl=True,
    clone_xvector_only=True,
    preset_voices=True,
    reusable_clone_prompt=True,
)


def _is_out_of_memory(exc: BaseException) -> bool:
    """Return whether an exception reports exhausted memory."""
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


class TorchQwen3TtsEngine:
    """TtsEngine backed by qwen-tts; clone prompts are cached per voice version."""

    def __init__(self, *, device: str, conditioning_cache_size: int) -> None:
        """Start unloaded on the given device."""
        self._device = device
        self._model: Qwen3TTSModel | None = None
        self._checkpoint: InstalledModel | None = None
        self._cache = VoiceConditioningCache(conditioning_cache_size)

    @property
    def name(self) -> TtsBackendName:
        """Return the runtime identifier."""
        return TtsBackendName.TORCH

    @property
    def capabilities(self) -> TtsCapabilities:
        """Return qwen-tts features."""
        return CAPABILITIES

    @property
    def device(self) -> str:
        """Return the device the runtime runs on."""
        return self._device

    def load(self, checkpoint: InstalledModel) -> None:
        """Load a local checkpoint: bfloat16 on CUDA, float32 on CPU."""
        self.unload()
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        device_map = "cuda:0" if self._device == "cuda" else "cpu"
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                str(checkpoint.local_path), device_map=device_map, dtype=dtype, local_files_only=True
            )
        except (RuntimeError, OSError, ValueError) as exc:
            raise InferenceError(
                f"could not load TTS checkpoint {checkpoint.name}", retryable=not _is_out_of_memory(exc)
            ) from exc
        self._checkpoint = checkpoint

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        """Synthesize one chunk with a cloned or preset voice."""
        model, checkpoint = self._require_loaded()
        try:
            with torch.inference_mode():
                match request.voice:
                    case CloneReference() as reference:
                        wavs, sample_rate = self._clone(model, checkpoint, reference, request)
                    case PresetSpeaker(name=speaker):
                        if checkpoint.variant is not TtsVariant.CUSTOM_VOICE:
                            raise InvalidRequestError("preset speakers require a CustomVoice checkpoint")
                        wavs, sample_rate = model.generate_custom_voice(
                            text=request.text, speaker=speaker, language=request.language
                        )
        except (RuntimeError, ValueError) as exc:
            raise InferenceError("TTS inference failed", retryable=not _is_out_of_memory(exc)) from exc
        if not wavs:
            raise InferenceError("TTS returned no audio")
        samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return SynthesizedAudio(samples=samples, sample_rate=int(sample_rate))

    def preset_speakers(self, checkpoint: InstalledModel) -> tuple[str, ...]:
        """Return speaker names from the checkpoint's config.json without loading weights."""
        if checkpoint.variant is not TtsVariant.CUSTOM_VOICE:
            return ()
        config = json.loads((checkpoint.local_path / "config.json").read_text(encoding="utf-8"))
        speakers = (config.get("talker_config") or {}).get("spk_id") or {}
        return tuple(sorted(str(name) for name in speakers))

    def unload(self) -> None:
        """Release the checkpoint and its conditioning cache."""
        if self._model is None:
            return
        self._model = None
        self._checkpoint = None
        self._cache.clear()
        gc.collect()
        if self._device == "cuda":
            torch.cuda.empty_cache()

    def _clone(
        self, model: Qwen3TTSModel, checkpoint: InstalledModel, reference: CloneReference, request: SynthesisRequest
    ) -> tuple[list[Any], int]:
        """Synthesize with a cached clone prompt built from the pinned reference."""
        if checkpoint.variant is not TtsVariant.BASE:
            raise InvalidRequestError("cloned voices require a Base checkpoint")
        key = (reference.voice_version_id, str(TtsBackendName.TORCH), checkpoint.name)
        prompt = self._cache.get_or_create(
            key,
            lambda: model.create_voice_clone_prompt(
                ref_audio=str(reference.audio_path),
                ref_text=reference.text,
                x_vector_only_mode=reference.text is None,
            ),
        )
        wavs, sample_rate = model.generate_voice_clone(
            text=request.text, language=request.language, voice_clone_prompt=prompt
        )
        return list(wavs), int(sample_rate)

    def _require_loaded(self) -> tuple[Qwen3TTSModel, InstalledModel]:
        """Return the loaded model and checkpoint or raise InferenceError."""
        if self._model is None or self._checkpoint is None:
            raise InferenceError("no TTS checkpoint loaded")
        return self._model, self._checkpoint
