"""Builds the selected TTS runtime; runtime modules are imported lazily because they are optional extras."""

import contextlib
import sys
from typing import Literal

from app.domain.tts import TtsBackendName
from app.ports.tts import TtsEngine


def build_tts_engine(
    name: TtsBackendName, *, device: Literal["cuda", "cpu"], conditioning_cache_size: int
) -> TtsEngine:
    """Return the engine for name; imports run with stdout sent to stderr because qwen-tts prints import banners."""
    with contextlib.redirect_stdout(sys.stderr):
        if name is TtsBackendName.MLX:
            from app.adapters.tts.qwen_mlx.engine import MlxQwen3TtsEngine

            return MlxQwen3TtsEngine()
        from app.adapters.tts.qwen_torch.engine import TorchQwen3TtsEngine

        return TorchQwen3TtsEngine(device=device, conditioning_cache_size=conditioning_cache_size)
