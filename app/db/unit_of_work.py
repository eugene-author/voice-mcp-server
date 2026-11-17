"""SQLAlchemy unit of work: repositories sharing one session and transaction."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories.assets import SqlAssetRepository
from app.db.repositories.events import SqlJobEventRepository
from app.db.repositories.jobs import SqlJobRepository
from app.db.repositories.metrics import SqlQualityMetricsRepository
from app.db.repositories.transcripts import SqlTranscriptRepository
from app.db.repositories.voices import SqlVoiceRepository
from app.db.repositories.workers import SqlWorkerRepository
from app.domain.errors import PersistenceUnavailableError


class SqlUnitOfWork:
    """SQLAlchemy repositories sharing one session and transaction."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind all repositories to the given session."""
        self.jobs = SqlJobRepository(session)
        self.events = SqlJobEventRepository(session)
        self.assets = SqlAssetRepository(session)
        self.transcripts = SqlTranscriptRepository(session)
        self.metrics = SqlQualityMetricsRepository(session)
        self.voices = SqlVoiceRepository(session)
        self.workers = SqlWorkerRepository(session)


class SqlUnitOfWorkFactory:
    """Creates a transactional SqlUnitOfWork per operation."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """Store the session factory built in the composition root."""
        self._session_factory = session_factory

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[SqlUnitOfWork]:
        """Yield a unit of work that commits on success and rolls back on any exception."""
        try:
            async with self._session_factory() as session, session.begin():
                yield SqlUnitOfWork(session)
        except (OperationalError, InterfaceError, OSError, ConnectionError) as exc:
            raise PersistenceUnavailableError("the database is unavailable") from exc
        except DBAPIError as exc:
            if exc.connection_invalidated:
                raise PersistenceUnavailableError("the database connection was lost") from exc
            raise
