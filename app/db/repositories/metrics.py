"""SQLAlchemy quality metrics repository."""

from datetime import datetime
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import QualityMetricsRow
from app.domain.ids import JobId


class SqlQualityMetricsRepository:
    """QualityMetricsRepository over quality_metrics."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to a session owned by the unit of work."""
        self._session = session

    async def add(self, job_id: JobId, metrics: dict[str, Any], *, created_at: datetime) -> None:
        """Store or replace the metrics of one job."""
        statement = insert(QualityMetricsRow).values(
            job_id=job_id, metrics=metrics, wer=metrics.get("wer"), cer=metrics.get("cer"), created_at=created_at
        )
        await self._session.execute(
            statement.on_conflict_do_update(
                index_elements=["job_id"], set_={"metrics": statement.excluded.metrics, "created_at": created_at}
            )
        )
