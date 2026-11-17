"""Speech-only timeline that maps concatenated speech audio back to source time."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from app.domain.stt import Segment, TranscriptionResult, Word


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    """Speech region in source seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        """Return the region length in seconds."""
        return self.end - self.start


class SpeechTimeline:
    """Merged speech regions and the mapping between speech-only and source time."""

    def __init__(self, regions: Sequence[SpeechRegion], *, merge_gap_seconds: float = 0.0) -> None:
        """Sort, clip and merge regions separated by less than merge_gap_seconds."""
        self._regions = self._merge(regions, merge_gap_seconds)
        offsets: list[float] = []
        total = 0.0
        for region in self._regions:
            offsets.append(total)
            total += region.duration
        self._offsets = tuple(offsets)
        self._speech_duration = total

    @property
    def regions(self) -> tuple[SpeechRegion, ...]:
        """Return the merged regions in source order."""
        return self._regions

    @property
    def speech_duration(self) -> float:
        """Return the total speech duration in seconds."""
        return self._speech_duration

    @property
    def is_empty(self) -> bool:
        """Return whether no speech was detected."""
        return not self._regions

    def to_source(self, speech_time: float) -> float:
        """Map a timestamp in the concatenated speech audio to the source timeline."""
        if not self._regions:
            return speech_time
        clamped = min(max(speech_time, 0.0), self._speech_duration)
        for region, offset in zip(reversed(self._regions), reversed(self._offsets), strict=True):
            if clamped >= offset:
                return region.start + min(clamped - offset, region.duration)
        return self._regions[0].start

    def extract(self, samples: npt.NDArray[np.float32], sample_rate: int) -> npt.NDArray[np.float32]:
        """Return the speech regions of samples concatenated into one array."""
        if not self._regions:
            return np.zeros(0, dtype=np.float32)
        parts = [samples[int(region.start * sample_rate) : int(region.end * sample_rate)] for region in self._regions]
        merged: npt.NDArray[np.float32] = np.concatenate(parts).astype(np.float32, copy=False)
        return merged

    @staticmethod
    def _merge(regions: Sequence[SpeechRegion], merge_gap_seconds: float) -> tuple[SpeechRegion, ...]:
        """Return sorted, non-overlapping regions with small gaps merged."""
        merged: list[SpeechRegion] = []
        for region in sorted((r for r in regions if r.end > r.start), key=lambda r: r.start):
            if merged and region.start - merged[-1].end < merge_gap_seconds:
                last = merged[-1]
                merged[-1] = SpeechRegion(start=last.start, end=max(last.end, region.end))
            else:
                merged.append(SpeechRegion(start=max(region.start, 0.0), end=region.end))
        return tuple(merged)


def map_to_source(result: TranscriptionResult, timeline: SpeechTimeline) -> TranscriptionResult:
    """Return the result with speech-only segment and word timestamps mapped to the source timeline."""

    def word(item: Word) -> Word:
        """Map one word."""
        return replace(item, start=timeline.to_source(item.start), end=timeline.to_source(item.end))

    def segment(item: Segment) -> Segment:
        """Map one segment and its words."""
        return replace(
            item,
            start=timeline.to_source(item.start),
            end=timeline.to_source(item.end),
            words=tuple(word(w) for w in item.words),
        )

    return replace(result, segments=tuple(segment(s) for s in result.segments))
