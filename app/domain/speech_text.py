"""Language codes, text normalization and sentence chunking for synthesis, and transcript similarity."""

import difflib
import re
import unicodedata

from app.domain.errors import InvalidRequestError

_SENTENCE_END = re.compile("(?<=[.!?;:\u3002\uff01\uff1f])\\s+")
_SOFT_BREAK = re.compile("(?<=[,\u3001\uff0c])\\s+")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[\w']+", re.UNICODE)
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")


def normalize_text(text: str) -> str:
    """Return NFC-normalized text with collapsed whitespace and no control characters."""
    normalized = unicodedata.normalize("NFC", text)
    cleaned = "".join(ch for ch in normalized if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return _WHITESPACE.sub(" ", cleaned).strip()


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Split a sentence longer than max_chars at commas, then at word boundaries."""
    pieces: list[str] = []
    for clause in _SOFT_BREAK.split(sentence):
        while len(clause) > max_chars:
            cut = clause.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(clause[:cut].strip())
            clause = clause[cut:].strip()
        if clause:
            pieces.append(clause)
    return pieces


def chunk_text(text: str, *, max_chars: int) -> list[str]:
    """Return sentence-aligned chunks of at most max_chars that together contain every word of text."""
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(normalize_text(text)):
        for piece in _split_long(sentence, max_chars) if len(sentence) > max_chars else [sentence]:
            candidate = f"{current} {piece}".strip()
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def words_of(text: str) -> list[str]:
    """Return lowercase word tokens."""
    return [token.lower() for token in _WORD.findall(unicodedata.normalize("NFKC", text))]


def transcript_similarity(expected: str, recognized: str) -> float:
    """Return a 0-1 word-sequence similarity between an expected and a recognized transcript."""
    expected_words, recognized_words = words_of(expected), words_of(recognized)
    if not expected_words and not recognized_words:
        return 1.0
    return difflib.SequenceMatcher(a=expected_words, b=recognized_words, autojunk=False).ratio()


def normalize_language(language: str | None) -> str | None:
    """Return a lowercase language code, None for auto, or raise for malformed codes."""
    if language is None or language.strip().lower() in {"", "auto"}:
        return None
    code = language.strip()
    if not _LANGUAGE_CODE.match(code):
        raise InvalidRequestError("language must be an ISO 639 code such as en or pt-BR, or auto")
    return code.split("-")[0].lower()
