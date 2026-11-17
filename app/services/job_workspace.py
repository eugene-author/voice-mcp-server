"""Job workspace: one directory per claim for a job's work and staged files, so no two claims share a directory."""

import contextlib
import shutil
from pathlib import Path

from app.domain.ids import JobId
from app.domain.jobs import ClaimedJob


class JobWorkspace:
    """Creates and removes working directories under the workspace root."""

    def __init__(self, root: Path) -> None:
        """Create the workspace root with restrictive permissions."""
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def job_dir(self, job_id: JobId) -> Path:
        """Return a working directory for one job, creating it if needed; a claim uses claim_dir instead."""
        path = self._root / "jobs" / str(job_id)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path

    def staged_dir(self, job_id: JobId) -> Path:
        """Return the directory holding a batch parent's finalized outputs awaiting publication."""
        path = self.job_dir(job_id) / "staged"
        path.mkdir(exist_ok=True, mode=0o700)
        return path

    def claim_dir(self, job: ClaimedJob) -> Path:
        """Return this claim's private working directory, creating it if needed."""
        path = self._claims(job.id) / str(job.claim_token)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path

    def claim_staged_dir(self, job: ClaimedJob) -> Path:
        """Return the directory holding this claim's finalized outputs awaiting publication."""
        path = self.claim_dir(job) / "staged"
        path.mkdir(exist_ok=True, mode=0o700)
        return path

    def other_claim_dirs(self, job: ClaimedJob) -> list[Path]:
        """Return the working directories other claims of this job left behind."""
        own = self.claim_dir(job)
        return [path for path in self._claims(job.id).iterdir() if path != own]

    def remove_claim(self, job: ClaimedJob) -> None:
        """Delete this claim's directory only, and the job's directories if nothing else is left in them."""
        claims = self._claims(job.id)
        shutil.rmtree(claims / str(job.claim_token), ignore_errors=True)
        with contextlib.suppress(OSError):
            claims.rmdir()
            claims.parent.rmdir()

    def remove_job(self, job_id: JobId) -> None:
        """Delete everything of a finished job, which only its accepted claim or batch aggregation may do."""
        shutil.rmtree(self._root / "jobs" / str(job_id), ignore_errors=True)

    def _claims(self, job_id: JobId) -> Path:
        """Return the directory holding one directory per claim of a job."""
        return self._root / "jobs" / str(job_id) / "claims"
