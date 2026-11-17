"""SQLAlchemy job event repository."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import JobEventRow
from app.domain.jobs import JobEvent


class SqlJobEventRepository:
    """Append-only job history in job_events."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to a session owned by the unit of work."""
        self._session = session

    async def add(self, event: JobEvent) -> None:
        """Append one event."""
        self._session.add(
            JobEventRow(
                job_id=event.job_id,
                at=event.at,
                event_type=event.event_type,
                phase=event.phase.value if event.phase else None,
                message=event.message,
                event_metadata=event.metadata,
            )
        )
