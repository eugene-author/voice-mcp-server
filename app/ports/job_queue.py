"""Job wake-up queue and inline-wait ports."""

from typing import Protocol

from app.domain.ids import JobId
from app.domain.jobs import JobStatus


class JobQueue(Protocol):
    """Wake-up hints for the worker; PostgreSQL stays the source of truth."""

    def notify(self, job_id: JobId) -> None:
        """Wake the worker for a committed job without blocking."""
        ...

    async def get(self) -> JobId:
        """Wait for the next woken job ID."""
        ...

    def task_done(self) -> None:
        """Mark the last job ID returned by get as processed."""
        ...


class JobWaiters(Protocol):
    """In-process futures that let tool calls wait for a job briefly."""

    async def wait(self, job_id: JobId, *, timeout_seconds: float) -> bool:
        """Wait until the job is resolved or the timeout passes; return whether it was resolved."""
        ...

    def resolve_if_terminal(self, job_id: JobId, status: JobStatus) -> None:
        """Release waiters of a job that reached a terminal status."""
        ...
