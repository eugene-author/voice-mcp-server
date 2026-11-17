"""Fixed-region voice activity detector."""

import numpy as np
import numpy.typing as npt

from app.domain.stt import VadSettings
from app.domain.timeline import SpeechRegion


class MockVoiceActivityDetector:
    """Returns configured regions."""

    def __init__(self, regions: tuple[SpeechRegion, ...] = (SpeechRegion(0.5, 1.5),)) -> None:
        """Store the regions to report."""
        self.regions = regions
        self.calls = 0

    def detect(
        self, samples: npt.NDArray[np.float32], sample_rate: int, settings: VadSettings
    ) -> tuple[SpeechRegion, ...]:
        """Return the configured regions."""
        self.calls += 1
        return self.regions
