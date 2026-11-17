"""Word and character error rates."""

from collections.abc import Sequence

from app.domain.speech_text import normalize_text, words_of


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return the Levenshtein distance between two token sequences."""
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for column, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (ref_token != hyp_token))
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return WER of hypothesis against reference over normalized lowercase words."""
    ref_words = words_of(reference)
    if not ref_words:
        return 0.0 if not words_of(hypothesis) else 1.0
    return _edit_distance(ref_words, words_of(hypothesis)) / len(ref_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Return CER of hypothesis against reference over normalized lowercase characters without spaces."""
    ref_chars = list(normalize_text(" ".join(words_of(reference))).replace(" ", ""))
    hyp_chars = list(normalize_text(" ".join(words_of(hypothesis))).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)
