"""Unit of work ports that bundle the repositories in one transaction."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from app.ports.repositories import (
    AssetRepository,
    JobEventRepository,
    JobRepository,
    QualityMetricsRepository,
    TranscriptRepository,
    VoiceRepository,
    WorkerRepository,
)


class UnitOfWork(Protocol):
    """Repositories sharing one transaction."""

    @property
    def jobs(self) -> JobRepository:
        """Return the job repository."""
        ...

    @property
    def events(self) -> JobEventRepository:
        """Return the job event repository."""
        ...

    @property
    def assets(self) -> AssetRepository:
        """Return the asset repository."""
        ...

    @property
    def transcripts(self) -> TranscriptRepository:
        """Return the transcript repository."""
        ...

    @property
    def metrics(self) -> QualityMetricsRepository:
        """Return the quality metrics repository."""
        ...

    @property
    def voices(self) -> VoiceRepository:
        """Return the voice repository."""
        ...

    @property
    def workers(self) -> WorkerRepository:
        """Return the worker registry repository."""
        ...


class UnitOfWorkFactory(Protocol):
    """Opens one transactional unit of work per operation."""

    def __call__(self) -> AbstractAsyncContextManager[UnitOfWork]:
        """Return a context that commits on success and rolls back on any exception."""
        ...
