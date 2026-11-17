"""Pure domain rules: timeline mapping, asset keys, quality, subtitles, text and error rates."""

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from app.domain.assets import AssetKey, job_input_key, job_output_key, voice_reference_key
from app.domain.audio import clipped_fraction, stitch
from app.domain.caller import Caller
from app.domain.errors import InvalidRequestError
from app.domain.evaluation import char_error_rate, word_error_rate
from app.domain.ids import JobId, VoiceId, new_claim_token, uuid7
from app.domain.jobs import JobCursor
from app.domain.quality import annotate, flag_segments, remove_fillers, remove_repetitions
from app.domain.speech_text import chunk_text, transcript_similarity
from app.domain.stt import Segment, TranscriptionResult, Word
from app.domain.subtitles import render_json, render_srt, render_vtt, transcript_from_document
from app.domain.timeline import SpeechRegion, SpeechTimeline

OWNER = Caller("https://issuer", "alice")


def test_timeline_merges_small_gaps_and_maps_back() -> None:
    """Merge regions closer than the gap and map speech time to source time."""
    timeline = SpeechTimeline(
        [SpeechRegion(5.0, 6.0), SpeechRegion(1.0, 2.0), SpeechRegion(2.1, 3.0)], merge_gap_seconds=0.3
    )
    assert timeline.regions == (SpeechRegion(1.0, 3.0), SpeechRegion(5.0, 6.0))
    assert timeline.speech_duration == pytest.approx(3.0)
    assert timeline.to_source(0.5) == pytest.approx(1.5)
    assert timeline.to_source(2.5) == pytest.approx(5.5)
    assert timeline.to_source(99) == pytest.approx(6.0)
    assert timeline.to_source(-1) == pytest.approx(1.0)


def test_timeline_extract_and_empty() -> None:
    """Concatenate speech samples and handle the no-speech case."""
    samples = np.arange(10, dtype=np.float32)
    timeline = SpeechTimeline([SpeechRegion(0.2, 0.4), SpeechRegion(0.7, 0.9)])
    assert timeline.extract(samples, 10).tolist() == [2, 3, 7, 8]
    empty = SpeechTimeline([])
    assert empty.is_empty
    assert empty.extract(samples, 10).size == 0
    assert empty.to_source(1.5) == 1.5


def test_asset_keys_use_opaque_owner_and_layout() -> None:
    """Build the documented key layout with an opaque owner key."""
    job_id = JobId(uuid7())
    key = job_output_key(OWNER, job_id, "transcript.txt")
    assert key.value == f"users/{OWNER.owner_key}/jobs/{job_id}/outputs/transcript.txt"
    assert "alice" not in key.value
    assert len(OWNER.owner_key) == 32
    assert job_input_key(OWNER, job_id, "MP3").value.endswith("/inputs/original.mp3")
    token = new_claim_token()
    assert job_output_key(OWNER, job_id, "a.txt", claim_token=token).value.endswith(f"/outputs/{token}/a.txt")
    assert job_input_key(OWNER, job_id, "wav", claim_token=token).value.endswith(f"/inputs/{token}/original.wav")
    assert voice_reference_key(OWNER, VoiceId("voice_abc"), 2, "wav", claim_token=token).value.endswith(
        f"/voices/voice_abc/v2/{token}/reference.wav"
    )


@pytest.mark.parametrize("bad", ["../etc/passwd", "/abs/path", "a/../b", "a//b", ".hidden/x"])
def test_asset_keys_reject_traversal(bad: str) -> None:
    """Reject keys that could escape the storage root."""
    with pytest.raises(InvalidRequestError):
        AssetKey(bad)


@pytest.mark.parametrize("name", ["..", "../x", "a/b", ""])
def test_output_names_reject_traversal(name: str) -> None:
    """Reject output names that are not single safe components."""
    with pytest.raises(InvalidRequestError):
        job_output_key(OWNER, JobId(uuid7()), name)


def test_uuid7_is_time_ordered_and_versioned() -> None:
    """Generate version 7 UUIDs that sort by creation time."""
    first, second = uuid7(), uuid7()
    assert first.version == 7
    assert first.variant == "specified in RFC 4122"
    assert first.bytes[:6] <= second.bytes[:6]


def test_cursor_round_trip_and_rejection() -> None:
    """Encode and decode cursors and reject garbage."""
    cursor = JobCursor(datetime(2026, 1, 1, tzinfo=UTC), JobId(uuid7()))
    assert JobCursor.decode(cursor.encode()) == cursor
    with pytest.raises(InvalidRequestError):
        JobCursor.decode("not-a-cursor")


def _segment(text: str, **signals: float) -> Segment:
    """Return a segment with optional quality signals."""
    return Segment(start=0.0, end=1.0, text=text, **signals)


def test_quality_flags() -> None:
    """Flag text in silence, high compression and repetition; skip missing signals."""
    flags = flag_segments(
        [
            _segment("Thanks for watching!", no_speech_prob=0.9, avg_logprob=-1.5),
            _segment("ok", compression_ratio=3.0),
            _segment("the the the the end"),
            _segment("fine", no_speech_prob=None),
        ]
    )
    assert [(f.segment_index, f.kind) for f in flags] == [
        (0, "text_in_silence"),
        (1, "high_compression"),
        (2, "repetition"),
    ]


def test_filler_annotation_and_removal() -> None:
    """Annotate fillers and repeated starts; remove only on request."""
    result = TranscriptionResult(
        segments=(_segment("Um so we start"), _segment("So we start again"), _segment("So we start again")),
        language="en",
        language_probability=None,
        duration_seconds=3.0,
    )
    annotations = annotate(result.segments, frozenset({"um"}))
    assert ("filler", "um") in {(a.kind, a.text) for a in annotations}
    assert any(a.kind == "repeated_start" for a in annotations)
    assert remove_fillers(result, frozenset({"um"})).segments[0].text == "so we start"
    assert len(remove_repetitions(result).segments) == 2


def test_subtitle_rendering_and_json_round_trip() -> None:
    """Render SRT/VTT timestamps and rebuild results from JSON."""
    result = TranscriptionResult(
        segments=(
            Segment(start=1.5, end=3661.25, text=" Hello ", words=(Word("Hello", 1.5, 2.0, 0.9),)),
            Segment(start=4.0, end=5.0, text="   "),
        ),
        language="en",
        language_probability=0.9,
        duration_seconds=3700,
    )
    assert render_srt(result) == "1\n00:00:01,500 --> 01:01:01,250\nHello\n"
    assert render_vtt(result).startswith("WEBVTT\n\n00:00:01.500 --> 01:01:01.250\nHello")
    rebuilt = transcript_from_document(json.loads(render_json(result)))
    assert rebuilt.segments[0].words[0].text == "Hello"
    assert rebuilt.language == "en"


def test_chunk_text_keeps_every_word_within_limit() -> None:
    """Split on sentences, then commas and words, without exceeding the limit or dropping words."""
    text = "First sentence here. " + ("word, " * 60) + "Last one!"
    chunks = chunk_text(text, max_chars=60)
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert " ".join(chunks).split() == " ".join(text.split()).split()
    assert chunk_text("Привет! Как дела?", max_chars=300) == ["Привет! Как дела?"]


def test_transcript_similarity() -> None:
    """Score matching transcripts high and unrelated ones low."""
    assert transcript_similarity("Hello, it's nice to meet you.", "hello its nice to meet you") > 0.8
    assert transcript_similarity("Hello world", "completely different words here") < 0.2


def test_error_rates() -> None:
    """Compute WER and CER over normalized tokens."""
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert char_error_rate("abc", "abd") == pytest.approx(1 / 3)
    assert word_error_rate("", "") == 0.0


def test_stitch_inserts_gaps_and_fades() -> None:
    """Concatenate chunks with silent gaps and faded edges."""
    chunk = np.ones(1000, dtype=np.float32)
    stitched = stitch([chunk, chunk], sample_rate=1000, gap_ms=100)
    assert stitched.size == 2100
    assert stitched[0] == 0.0
    assert stitched[1050] == 0.0
    assert clipped_fraction(np.array([1.0, 0.0], dtype=np.float32)) == 0.5
