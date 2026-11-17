"""Allowlist validation of host paths supplied by callers."""

from collections.abc import Iterable
from pathlib import Path

from app.domain.errors import InvalidRequestError

_OUTSIDE_MESSAGE = "path is not inside an allowed input directory"


class InputPathPolicy:
    """Resolves caller paths and accepts only files inside the allowed input directories."""

    def __init__(
        self, *, allowed_directories: Iterable[Path], accepted_formats: Iterable[str], max_input_bytes: int
    ) -> None:
        """Store resolved allowlisted directories, accepted extensions and the size limit."""
        self._allowed = tuple(Path(directory).resolve() for directory in allowed_directories)
        self._formats = frozenset(fmt.lower().lstrip(".") for fmt in accepted_formats)
        self._max_input_bytes = max_input_bytes

    @property
    def allowed_directories(self) -> tuple[Path, ...]:
        """Return the allowlisted directories."""
        return self._allowed

    def resolve_file(self, raw_path: Path | str) -> Path:
        """Return the real path of an allowed, accepted, size-limited audio file."""
        path = self._resolve_inside(raw_path)
        if not path.is_file():
            raise InvalidRequestError("file does not exist or is not a regular file")
        if path.suffix.lower().lstrip(".") not in self._formats:
            raise InvalidRequestError(f"unsupported audio format; accepted: {', '.join(sorted(self._formats))}")
        self._require_usable_size(path, label="file")
        return path

    def resolve_directory(self, raw_path: Path | str) -> Path:
        """Return the real path of an allowed directory."""
        path = self._resolve_inside(raw_path)
        if not path.is_dir():
            raise InvalidRequestError("directory does not exist")
        return path

    def audio_files(self, directory: Path, *, recursive: bool, limit: int) -> list[Path]:
        """Walk lazily, stop past limit accepted files, skip symlink escapes, then size-check the sorted files."""
        pattern = "**/*" if recursive else "*"
        found: list[tuple[Path, Path]] = []
        for candidate in directory.glob(pattern):
            if candidate.suffix.lower().lstrip(".") not in self._formats or not candidate.is_file():
                continue
            real = candidate.resolve()
            if not self._is_allowed(real):
                continue
            found.append((candidate, real))
            if len(found) > limit:
                raise InvalidRequestError(f"directory contains more than {limit} audio files")
        files = [real for _, real in sorted(found)]
        for real in files:
            self._require_usable_size(real, label=real.name)
        return files

    def _require_usable_size(self, path: Path, *, label: str) -> None:
        """Reject an empty file or one over the size limit, calling it label in the error."""
        size = path.stat().st_size
        if size > self._max_input_bytes:
            raise InvalidRequestError(f"{label} exceeds the maximum input size")
        if size == 0:
            raise InvalidRequestError(f"{label} is empty")

    def _resolve_inside(self, raw_path: Path | str) -> Path:
        """Resolve symlinks and require the result inside an allowed directory."""
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            raise InvalidRequestError("path must be absolute")
        try:
            real = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise InvalidRequestError("path cannot be resolved") from exc
        if not self._is_allowed(real):
            raise InvalidRequestError(_OUTSIDE_MESSAGE)
        return real

    def _is_allowed(self, real_path: Path) -> bool:
        """Return whether a resolved path lies inside an allowlisted directory."""
        return any(real_path.is_relative_to(directory) for directory in self._allowed)
