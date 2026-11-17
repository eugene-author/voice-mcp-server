"""One claim's staged outputs and metadata with a .complete marker last; a retry copies them into its own claim."""

import json
import logging
import os
import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from app.domain.jobs import ClaimedJob
from app.services.job_workspace import JobWorkspace

logger = logging.getLogger(__name__)
COMPLETE_MARKER = ".complete"
METADATA_FILE = "outcome.json"

type CheckpointParser[T] = Callable[[Path, Mapping[str, Any]], T]
"""Builds a typed checkpoint from the staged directory and its metadata; raises KeyError, TypeError, ValueError or
OSError when they are unusable."""


class StagedJobAssets:
    """Manages one claim's staged outputs and checkpoint metadata; publishing and persistence happen elsewhere."""

    def __init__(self, workspace: JobWorkspace) -> None:
        """Use the job workspace's per-claim staged directories."""
        self._workspace = workspace

    def path(self, job: ClaimedJob, name: str) -> Path:
        """Return where a staged output file of this claim lives."""
        return self._workspace.claim_staged_dir(job) / name

    def write_text(self, job: ClaimedJob, name: str, text: str) -> Path:
        """Write a staged text output atomically and return its path."""
        target = self.path(job, name)
        _atomic_write(target, text)
        return target

    def load[T](self, job: ClaimedJob, *, required: Iterable[str], parse: CheckpointParser[T]) -> T | None:
        """Copy the newest usable checkpoint an earlier claim committed into this claim and return it, else None."""
        directory = self._workspace.claim_staged_dir(job)
        names = (*required, METADATA_FILE)
        for committed in self._committed_by_others(job):
            try:
                for name in names:
                    shutil.copy2(committed / name, directory / name)
                metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise TypeError("checkpoint metadata is not an object")
                checkpoint = parse(directory, metadata)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "skipping unusable staged checkpoint",
                    extra={"job_id": str(job.id), "reason": f"{type(exc).__name__}: {exc}"},
                )
                self.invalidate(job)
                continue
            _atomic_write(directory / COMPLETE_MARKER, "")
            logger.info("reusing staged outputs", extra={"job_id": str(job.id)})
            return checkpoint
        return None

    def commit(self, job: ClaimedJob, metadata: Mapping[str, Any], *, required: Iterable[str]) -> None:
        """Record metadata and mark the checkpoint complete once every required output exists."""
        directory = self._workspace.claim_staged_dir(job)
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"cannot commit staged outputs; missing: {', '.join(missing)}")
        _atomic_write(directory / METADATA_FILE, json.dumps(metadata))
        _atomic_write(directory / COMPLETE_MARKER, "")

    def invalidate(self, job: ClaimedJob) -> None:
        """Remove this claim's marker first, then its staged files, so a partial cleanup cannot read as a checkpoint."""
        directory = self._workspace.claim_staged_dir(job)
        (directory / COMPLETE_MARKER).unlink(missing_ok=True)
        for entry in directory.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)

    def _committed_by_others(self, job: ClaimedJob) -> list[Path]:
        """Return the staged directories of other claims with a complete marker, newest commit first."""
        markers = [path / "staged" / COMPLETE_MARKER for path in self._workspace.other_claim_dirs(job)]
        found = [marker for marker in markers if marker.is_file()]
        return [marker.parent for marker in sorted(found, key=lambda marker: marker.stat().st_mtime_ns, reverse=True)]


def _atomic_write(target: Path, text: str) -> None:
    """Write text to a temporary sibling, flush it to disk, and rename it over target."""
    partial = target.with_name(target.name + ".part")
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(target)
