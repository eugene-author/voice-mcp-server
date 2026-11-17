"""In-memory unit of work over the repository doubles."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy

from app.domain.errors import PersistenceUnavailableError
from tests.mocks.repositories import (
    Database,
    MockAssetRepository,
    MockJobEventRepository,
    MockJobRepository,
    MockQualityMetricsRepository,
    MockTranscriptRepository,
    MockVoiceRepository,
    MockWorkerRepository,
)


class MockUnitOfWork:
    """Repositories over one in-memory database."""

    def __init__(self, db: Database) -> None:
        """Bind every repository to the database."""
        self.jobs = MockJobRepository(db)
        self.events = MockJobEventRepository(db)
        self.assets = MockAssetRepository(db)
        self.transcripts = MockTranscriptRepository(db)
        self.metrics = MockQualityMetricsRepository(db)
        self.voices = MockVoiceRepository(db)
        self.workers = MockWorkerRepository(db)


class MockUnitOfWorkFactory:
    """Transactional in-memory unit of work: changes are discarded when the block raises."""

    def __init__(self) -> None:
        """Start with an empty database."""
        self.db = Database()
        self.available = True
        self.commits = 0

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[MockUnitOfWork]:
        """Yield a unit of work on a copy and commit it only on success."""
        if not self.available:
            raise PersistenceUnavailableError("the database is unavailable")
        working = deepcopy(self.db)
        yield MockUnitOfWork(working)
        self.db = working
        self.commits += 1
