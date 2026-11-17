"""Voice activity detection port."""

from typing import Protocol

import numpy as np
import numpy.typing as npt

from app.domain.stt import VadSettings
from app.domain.timeline import SpeechRegion


class VoiceActivityDetector(Protocol):
    """Blocking Silero VAD, always called on the inference executor thread."""

    def detect(
        self, samples: npt.NDArray[np.float32], sample_rate: int, settings: VadSettings
    ) -> tuple[SpeechRegion, ...]:
        """Return speech regions in seconds."""
        ...
