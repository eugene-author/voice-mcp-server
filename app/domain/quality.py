"""Transcript quality heuristics, filler annotation and repeated-start detection."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from app.domain.stt import Segment, TranscriptionResult

NO_SPEECH_PROB_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4
MIN_REPEATED_PHRASE_COUNT = 3
_WORD = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SegmentFlag:
    """Heuristic quality flag raised on one segment."""

    segment_index: int
    kind: str
    """One of text_in_silence, low_confidence, high_compression, repetition."""

    detail: str


@dataclass(frozen=True, slots=True)
class Annotation:
    """Filler word or repeated sentence start found in the transcript."""

    segment_index: int
    kind: str
    """Either filler or repeated_start."""

    text: str


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Heuristic flags and annotations for one transcript."""

    flags: tuple[SegmentFlag, ...] = ()
    annotations: tuple[Annotation, ...] = ()

    def warnings(self) -> tuple[str, ...]:
        """Return a short summary suitable for job warnings."""
        if not self.flags:
            return ()
        kinds: dict[str, int] = {}
        for flag in self.flags:
            kinds[flag.kind] = kinds.get(flag.kind, 0) + 1
        summary = ", ".join(f"{kind} x{count}" for kind, count in sorted(kinds.items()))
        return (f"possible transcription issues (heuristic): {summary}",)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report."""
        return {
            "flags": [{"segment": f.segment_index, "kind": f.kind, "detail": f.detail} for f in self.flags],
            "annotations": [{"segment": a.segment_index, "kind": a.kind, "text": a.text} for a in self.annotations],
        }


def _words(text: str) -> list[str]:
    """Return lowercase word tokens."""
    return [token.lower() for token in _WORD.findall(text)]


def _has_repetition(text: str) -> bool:
    """Return whether a 1-4 word phrase repeats consecutively at least MIN_REPEATED_PHRASE_COUNT times."""
    tokens = _words(text)
    for size in range(1, 5):
        for start in range(len(tokens) - size * MIN_REPEATED_PHRASE_COUNT + 1):
            phrase = tokens[start : start + size]
            repeats = 1
            position = start + size
            while tokens[position : position + size] == phrase:
                repeats += 1
                position += size
            if repeats >= MIN_REPEATED_PHRASE_COUNT:
                return True
    return False


def flag_segments(segments: Sequence[Segment]) -> tuple[SegmentFlag, ...]:
    """Return quality flags per segment; missing signals are skipped, not guessed."""
    flags: list[SegmentFlag] = []
    for index, segment in enumerate(segments):
        no_speech, logprob, ratio = segment.no_speech_prob, segment.avg_logprob, segment.compression_ratio
        if no_speech is not None and logprob is not None:
            if no_speech > NO_SPEECH_PROB_THRESHOLD and logprob < AVG_LOGPROB_THRESHOLD:
                flags.append(SegmentFlag(index, "text_in_silence", f"no_speech_prob={no_speech:.2f}"))
        elif logprob is not None and logprob < AVG_LOGPROB_THRESHOLD:
            flags.append(SegmentFlag(index, "low_confidence", f"avg_logprob={logprob:.2f}"))
        if ratio is not None and ratio > COMPRESSION_RATIO_THRESHOLD:
            flags.append(SegmentFlag(index, "high_compression", f"compression_ratio={ratio:.2f}"))
        if _has_repetition(segment.text):
            flags.append(SegmentFlag(index, "repetition", "repeated phrase"))
    return tuple(flags)


def annotate(segments: Sequence[Segment], fillers: frozenset[str]) -> tuple[Annotation, ...]:
    """Return filler words and repeated sentence starts, without changing the text."""
    annotations: list[Annotation] = []
    previous_start: tuple[str, ...] = ()
    for index, segment in enumerate(segments):
        tokens = _words(segment.text)
        annotations.extend(Annotation(index, "filler", token) for token in tokens if token in fillers)
        start = tuple(tokens[:2])
        if len(start) == 2 and start == previous_start:
            annotations.append(Annotation(index, "repeated_start", " ".join(start)))
        previous_start = start
    return tuple(annotations)


def remove_fillers(result: TranscriptionResult, fillers: frozenset[str]) -> TranscriptionResult:
    """Return a copy of the transcript with filler words removed from segment text."""

    def clean(text: str) -> str:
        """Drop filler tokens and collapse whitespace."""
        kept = [word for word in text.split() if _words(word) and " ".join(_words(word)) not in fillers]
        return " ".join(kept)

    segments = tuple(replace(segment, text=clean(segment.text)) for segment in result.segments)
    return replace(result, segments=segments)


def remove_repetitions(result: TranscriptionResult) -> TranscriptionResult:
    """Return a copy without segments that repeat the previous segment's text exactly."""
    kept: list[Segment] = []
    for segment in result.segments:
        if kept and _words(kept[-1].text) == _words(segment.text):
            continue
        kept.append(segment)
    return replace(result, segments=tuple(kept))


def fillers_for(language: str | None, dictionary: Mapping[str, Sequence[str]]) -> frozenset[str]:
    """Return the filler set for a language code, empty when unknown."""
    if not language:
        return frozenset()
    return frozenset(word.lower() for word in dictionary.get(language.split("-")[0].lower(), ()))
