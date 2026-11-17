"""SQLAlchemy worker registry repository."""

from datetime import datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import WorkerRow
from app.domain.workers import CapabilitySnapshot, WorkerRecord, WorkerStatus


def _record(row: WorkerRow) -> WorkerRecord:
    """Map a workers row to a WorkerRecord."""
    return WorkerRecord(
        id=row.id,
        status=WorkerStatus(row.status),
        capabilities=CapabilitySnapshot.from_dict(row.capabilities),
        started_at=row.started_at,
        heartbeat_at=row.heartbeat_at,
        last_activity_at=row.last_activity_at,
        drain_requested_at=row.drain_requested_at,
        drained_at=row.drained_at,
        version=row.version,
    )


class SqlWorkerRepository:
    """WorkerRepository over the workers table."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to a session owned by the unit of work."""
        self._session = session

    async def register(self, record: WorkerRecord) -> None:
        """Insert or replace a worker's row."""
        values: dict[str, Any] = {
            "id": record.id,
            "status": record.status.value,
            "capabilities": record.capabilities.to_dict(),
            "started_at": record.started_at,
            "heartbeat_at": record.heartbeat_at,
            "last_activity_at": record.last_activity_at,
            "drain_requested_at": record.drain_requested_at,
            "drained_at": record.drained_at,
            "version": record.version,
        }
        updates = {key: value for key, value in values.items() if key != "id"}
        await self._session.execute(
            insert(WorkerRow).values(**values).on_conflict_do_update(index_elements=[WorkerRow.id], set_=updates)
        )

    async def beat(
        self,
        worker_id: str,
        *,
        status: WorkerStatus,
        now: datetime,
        last_activity_at: datetime | None,
        capabilities: CapabilitySnapshot | None = None,
    ) -> WorkerRecord | None:
        """Record status, heartbeat, activity and optionally capabilities; return the row, None if it is gone."""
        values: dict[str, Any] = {"status": status.value, "heartbeat_at": now, "last_activity_at": last_activity_at}
        if capabilities is not None:
            values["capabilities"] = capabilities.to_dict()
        row = await self._session.scalar(
            update(WorkerRow).where(WorkerRow.id == worker_id).values(**values).returning(WorkerRow)
        )
        return _record(row) if row is not None else None

    async def mark_stopped(self, worker_id: str, *, now: datetime) -> None:
        """Set status stopped with drained_at and a final heartbeat."""
        await self._session.execute(
            update(WorkerRow)
            .where(WorkerRow.id == worker_id)
            .values(status=WorkerStatus.STOPPED.value, drained_at=now, heartbeat_at=now)
        )

    async def get(self, worker_id: str) -> WorkerRecord | None:
        """Return one worker's row."""
        row = await self._session.get(WorkerRow, worker_id, populate_existing=True)
        return _record(row) if row is not None else None

    async def latest(self) -> WorkerRecord | None:
        """Return the row with the newest heartbeat, whatever its status."""
        row = await self._session.scalar(select(WorkerRow).order_by(WorkerRow.heartbeat_at.desc()).limit(1))
        return _record(row) if row is not None else None

    async def active(self, *, heartbeat_after: datetime) -> WorkerRecord | None:
        """Return the most recently started worker whose heartbeat is newer than heartbeat_after."""
        row = await self._session.scalar(
            select(WorkerRow)
            .where(WorkerRow.heartbeat_at > heartbeat_after, WorkerRow.status != WorkerStatus.STOPPED.value)
            .order_by(WorkerRow.started_at.desc())
            .limit(1)
        )
        return _record(row) if row is not None else None

    async def request_drain(self, worker_id: str, *, now: datetime) -> bool:
        """Set drain_requested_at if unset; return whether a row changed."""
        changed = await self._session.scalar(
            update(WorkerRow)
            .where(WorkerRow.id == worker_id, WorkerRow.drain_requested_at.is_(None))
            .values(drain_requested_at=now)
            .returning(WorkerRow.id)
        )
        return changed is not None

    async def cancel_drain(self, worker_id: str) -> bool:
        """Clear drain_requested_at; return whether a row changed."""
        changed = await self._session.scalar(
            update(WorkerRow)
            .where(WorkerRow.id == worker_id, WorkerRow.drain_requested_at.is_not(None))
            .values(drain_requested_at=None)
            .returning(WorkerRow.id)
        )
        return changed is not None

    async def delete_stopped(self, *, before: datetime) -> int:
        """Delete stopped rows older than before, always keeping the newest row; return the count."""
        newest = select(WorkerRow.id).order_by(WorkerRow.heartbeat_at.desc()).limit(1).scalar_subquery()
        result = await self._session.execute(
            delete(WorkerRow).where(
                WorkerRow.status == WorkerStatus.STOPPED.value, WorkerRow.heartbeat_at < before, WorkerRow.id != newest
            )
        )
        return int(getattr(result, "rowcount", 0) or 0)
