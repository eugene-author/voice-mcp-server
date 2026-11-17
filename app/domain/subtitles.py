"""Transcript rendering to TXT, JSON, SRT and VTT."""

import json
from typing import Any

from app.domain.stt import Segment, TranscriptionResult, Word


def _timestamp(seconds: float, separator: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm."""
    total_ms = max(round(seconds * 1000), 0)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def render_txt(result: TranscriptionResult) -> str:
    """Return the plain transcript text with a trailing newline."""
    return result.text + "\n"


def transcript_document(result: TranscriptionResult, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the JSON transcript document with segments and words."""
    document: dict[str, Any] = {
        "language": result.language,
        "language_probability": result.language_probability,
        "duration_seconds": result.duration_seconds,
        "text": result.text,
        "segments": [
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "no_speech_prob": segment.no_speech_prob,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "words": [
                    {
                        "text": word.text,
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "probability": word.probability,
                    }
                    for word in segment.words
                ],
            }
            for segment in result.segments
        ],
    }
    if extra:
        document.update(extra)
    return document


def render_json(result: TranscriptionResult, *, extra: dict[str, Any] | None = None) -> str:
    """Return the JSON transcript document as text."""
    return json.dumps(transcript_document(result, extra=extra), ensure_ascii=False, indent=2) + "\n"


def render_srt(result: TranscriptionResult) -> str:
    """Return SubRip subtitles."""
    blocks = [
        f"{index}\n{_timestamp(seg.start, ',')} --> {_timestamp(seg.end, ',')}\n{seg.text.strip()}\n"
        for index, seg in enumerate((s for s in result.segments if s.text.strip()), start=1)
    ]
    return "\n".join(blocks)


def render_vtt(result: TranscriptionResult) -> str:
    """Return WebVTT subtitles."""
    cues = [
        f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}\n{seg.text.strip()}\n"
        for seg in result.segments
        if seg.text.strip()
    ]
    return "WEBVTT\n\n" + "\n".join(cues)


def transcript_from_document(document: dict[str, Any]) -> TranscriptionResult:
    """Rebuild a transcription result from a JSON transcript document."""
    segments = tuple(
        Segment(
            start=float(item["start"]),
            end=float(item["end"]),
            text=str(item["text"]),
            words=tuple(
                Word(
                    text=str(word["text"]),
                    start=float(word["start"]),
                    end=float(word["end"]),
                    probability=word.get("probability"),
                )
                for word in item.get("words", ())
            ),
            no_speech_prob=item.get("no_speech_prob"),
            avg_logprob=item.get("avg_logprob"),
            compression_ratio=item.get("compression_ratio"),
        )
        for item in document.get("segments", ())
    )
    return TranscriptionResult(
        segments=segments,
        language=str(document.get("language") or ""),
        language_probability=document.get("language_probability"),
        duration_seconds=float(document.get("duration_seconds") or 0.0),
    )
