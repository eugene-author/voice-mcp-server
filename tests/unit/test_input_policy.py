"""Allowlisted input paths."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.domain.errors import InvalidRequestError
from app.services.input_policy import InputPathPolicy


@pytest.fixture
def allowed(tmp_path: Path) -> Path:
    """Return an allowed input directory."""
    directory = tmp_path / "inputs"
    directory.mkdir()
    return directory


def _policy(allowed: Path, max_bytes: int = 1_000_000) -> InputPathPolicy:
    """Return a policy for the allowed directory."""
    return InputPathPolicy(allowed_directories=[allowed], accepted_formats=["wav", "mp3"], max_input_bytes=max_bytes)


def test_accepts_allowed_file(allowed: Path) -> None:
    """Resolve a valid file inside the allowlist."""
    path = allowed / "a.wav"
    path.write_bytes(b"x")
    assert _policy(allowed).resolve_file(str(path)) == path.resolve()


@pytest.mark.parametrize("raw", ["/etc/passwd", "relative.wav"])
def test_rejects_outside_or_relative(allowed: Path, raw: str) -> None:
    """Reject paths outside the allowlist and relative paths."""
    with pytest.raises(InvalidRequestError):
        _policy(allowed).resolve_file(raw)


def test_rejects_symlink_escape(allowed: Path, tmp_path: Path) -> None:
    """Reject symlinks that resolve outside the allowlist."""
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"x")
    link = allowed / "link.wav"
    link.symlink_to(secret)
    with pytest.raises(InvalidRequestError, match="not inside"):
        _policy(allowed).resolve_file(str(link))


def test_rejects_dotdot_escape(allowed: Path, tmp_path: Path) -> None:
    """Reject .. traversal out of the allowlist."""
    (tmp_path / "x.wav").write_bytes(b"x")
    with pytest.raises(InvalidRequestError):
        _policy(allowed).resolve_file(str(allowed / ".." / "x.wav"))


def test_rejects_unsupported_format_large_and_empty(allowed: Path) -> None:
    """Enforce extension, size and emptiness limits."""
    (allowed / "a.txt").write_bytes(b"x")
    (allowed / "big.wav").write_bytes(b"x" * 20)
    (allowed / "empty.wav").write_bytes(b"")
    policy = _policy(allowed, max_bytes=10)
    for name in ("a.txt", "big.wav", "empty.wav", "missing.wav"):
        with pytest.raises(InvalidRequestError):
            policy.resolve_file(str(allowed / name))


def test_lists_audio_files_and_enforces_limit(allowed: Path, tmp_path: Path) -> None:
    """List accepted files, skip symlink escapes and enforce the item limit."""
    for name in ("b.wav", "a.mp3", "c.txt"):
        (allowed / name).write_bytes(b"x")
    (allowed / "sub").mkdir()
    (allowed / "sub" / "d.wav").write_bytes(b"x")
    outside = tmp_path / "o.wav"
    outside.write_bytes(b"x")
    (allowed / "escape.wav").symlink_to(outside)
    policy = _policy(allowed)
    assert [p.name for p in policy.audio_files(allowed, recursive=False, limit=10)] == ["a.mp3", "b.wav"]
    assert len(policy.audio_files(allowed, recursive=True, limit=10)) == 3
    with pytest.raises(InvalidRequestError):
        policy.audio_files(allowed, recursive=True, limit=2)


def test_scan_stops_once_the_limit_is_exceeded(allowed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop walking after limit + 1 accepted files instead of listing the whole tree first."""
    for index in range(40):
        (allowed / f"{index:02d}.wav").write_bytes(b"x")
    yielded: list[Path] = []
    original_glob = Path.glob

    def counting_glob(self: Path, pattern: str, **kwargs: Any) -> Iterator[Path]:
        """Yield the real glob results while counting how many were consumed."""
        for path in original_glob(self, pattern, **kwargs):
            yielded.append(path)
            yield path

    monkeypatch.setattr(Path, "glob", counting_glob)
    with pytest.raises(InvalidRequestError, match="more than 3"):
        _policy(allowed).audio_files(allowed, recursive=True, limit=3)
    assert len(yielded) == 4


def test_listing_is_sorted_by_path_whatever_the_walk_order(allowed: Path) -> None:
    """Return files ordered by path, including nested ones, with the same files as before."""
    for name in ("z.wav", "m.mp3", "a.wav"):
        (allowed / name).write_bytes(b"x")
    (allowed / "b").mkdir()
    (allowed / "b" / "c.wav").write_bytes(b"x")
    files = _policy(allowed).audio_files(allowed, recursive=True, limit=10)
    assert [str(path.relative_to(allowed.resolve())) for path in files] == ["a.wav", "b/c.wav", "m.mp3", "z.wav"]


def test_oversized_file_is_rejected_by_both_routes(allowed: Path) -> None:
    """Apply the same individual limit to a single file and a batch, naming the file in the batch error."""
    big = allowed / "big.wav"
    big.write_bytes(b"x" * 11)
    policy = _policy(allowed, max_bytes=10)
    with pytest.raises(InvalidRequestError, match="exceeds the maximum input size"):
        policy.resolve_file(str(big))
    with pytest.raises(InvalidRequestError, match=r"big\.wav exceeds the maximum input size"):
        policy.audio_files(allowed, recursive=False, limit=10)


def test_empty_file_is_rejected_in_a_batch(allowed: Path) -> None:
    """Reject an empty accepted-format file instead of queueing a job that cannot run."""
    (allowed / "fine.wav").write_bytes(b"x")
    (allowed / "empty.wav").write_bytes(b"")
    with pytest.raises(InvalidRequestError, match=r"empty\.wav is empty"):
        _policy(allowed).audio_files(allowed, recursive=False, limit=10)


def test_file_exactly_at_the_limit_is_accepted_by_both_routes(allowed: Path) -> None:
    """Treat the limit as inclusive on both routes."""
    exact = allowed / "exact.wav"
    exact.write_bytes(b"x" * 10)
    policy = _policy(allowed, max_bytes=10)
    assert policy.resolve_file(str(exact)) == exact.resolve()
    assert policy.audio_files(allowed, recursive=False, limit=10) == [exact.resolve()]


def test_batch_validation_follows_the_traversal(allowed: Path) -> None:
    """Check a nested file only when the walk is recursive."""
    (allowed / "fine.wav").write_bytes(b"x")
    (allowed / "sub").mkdir()
    (allowed / "sub" / "big.wav").write_bytes(b"x" * 11)
    policy = _policy(allowed, max_bytes=10)
    assert [p.name for p in policy.audio_files(allowed, recursive=False, limit=10)] == ["fine.wav"]
    with pytest.raises(InvalidRequestError, match=r"big\.wav exceeds"):
        policy.audio_files(allowed, recursive=True, limit=10)


def test_batch_still_filters_what_it_never_accepted(allowed: Path, tmp_path: Path) -> None:
    """Skip an oversized unsupported file and a symlink escaping to an oversized file, rather than failing the batch."""
    (allowed / "fine.wav").write_bytes(b"x")
    (allowed / "big.txt").write_bytes(b"x" * 11)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"x" * 11)
    (allowed / "escape.wav").symlink_to(outside)
    files = _policy(allowed, max_bytes=10).audio_files(allowed, recursive=False, limit=10)
    assert [p.name for p in files] == ["fine.wav"]


def test_symlink_inside_the_allowlist_is_validated_by_its_target(allowed: Path) -> None:
    """Measure the real file behind a symlink that stays inside the allowlist."""
    (allowed / "real").mkdir()
    (allowed / "real" / "big.mp3").write_bytes(b"x" * 11)
    (allowed / "link.wav").symlink_to(allowed / "real" / "big.mp3")
    with pytest.raises(InvalidRequestError, match=r"big\.mp3 exceeds"):
        _policy(allowed, max_bytes=10).audio_files(allowed, recursive=False, limit=10)


def test_item_limit_is_reported_before_file_sizes(allowed: Path) -> None:
    """Keep the batch item limit's error when a directory is also over it and holds an invalid file."""
    for name in ("a.wav", "b.wav", "c.wav"):
        (allowed / name).write_bytes(b"")
    with pytest.raises(InvalidRequestError, match="more than 2"):
        _policy(allowed).audio_files(allowed, recursive=False, limit=2)
