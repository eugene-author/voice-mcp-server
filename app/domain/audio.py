"""Pure sample-level audio helpers: stitching, clipping and level measurements."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FADE_MS = 10
CLIP_LEVEL = 0.999


def _faded(samples: npt.NDArray[np.float32], fade_samples: int) -> npt.NDArray[np.float32]:
    """Return a copy with short linear fades at both ends to avoid clicks."""
    result = samples.astype(np.float32, copy=True)
    length = min(fade_samples, len(result) // 2)
    if length > 0:
        ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
        result[:length] *= ramp
        result[-length:] *= ramp[::-1]
    return result


def stitch(chunks: Sequence[npt.NDArray[np.float32]], *, sample_rate: int, gap_ms: int) -> npt.NDArray[np.float32]:
    """Concatenate chunks with fades and fixed silent gaps between them."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    fade = sample_rate * FADE_MS // 1000
    gap = np.zeros(sample_rate * gap_ms // 1000, dtype=np.float32)
    parts: list[npt.NDArray[np.float32]] = []
    for index, chunk in enumerate(chunks):
        if index:
            parts.append(gap)
        parts.append(_faded(np.asarray(chunk, dtype=np.float32).reshape(-1), fade))
    return np.concatenate(parts)


def rms(samples: npt.NDArray[np.float32]) -> float:
    """Return the root-mean-square level."""
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0


def peak(samples: npt.NDArray[np.float32]) -> float:
    """Return the absolute peak level."""
    return float(np.max(np.abs(samples))) if len(samples) else 0.0


def clipped_fraction(samples: npt.NDArray[np.float32]) -> float:
    """Return the fraction of samples at or above full scale."""
    return float(np.mean(np.abs(samples) >= CLIP_LEVEL)) if len(samples) else 0.0
