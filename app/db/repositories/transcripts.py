"""SQLAlchemy transcript repository."""

from datetime import datetime

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TranscriptRow, TranscriptSegmentRow, TranscriptWordRow
from app.domain.caller import Caller
from app.domain.ids import JobId, uuid7
from app.domain.stt import TranscriptionResult


class SqlTranscriptRepository:
    """TranscriptRepository over transcripts, transcript_segments and transcript_words."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to a session owned by the unit of work."""
        self._session = session

    async def add(self, job_id: JobId, owner: Caller, result: TranscriptionResult, *, created_at: datetime) -> None:
        """Store a transcript with its segments and words using bulk inserts."""
        transcript_id = uuid7()
        self._session.add(
            TranscriptRow(
                id=transcript_id,
                job_id=job_id,
                owner_issuer=owner.issuer,
                owner_subject=owner.subject,
                language=result.language,
                language_probability=result.language_probability,
                duration_seconds=result.duration_seconds,
                text=result.text,
                created_at=created_at,
            )
        )
        await self._session.flush()
        if not result.segments:
            return
        segment_rows = [
            {
                "transcript_id": transcript_id,
                "position": position,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "no_speech_prob": segment.no_speech_prob,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
            }
            for position, segment in enumerate(result.segments)
        ]
        segment_ids = list(
            await self._session.scalars(
                insert(TranscriptSegmentRow).returning(TranscriptSegmentRow.id, sort_by_parameter_order=True),
                segment_rows,
            )
        )
        word_rows = [
            {
                "segment_id": segment_id,
                "position": position,
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "probability": word.probability,
            }
            for segment_id, segment in zip(segment_ids, result.segments, strict=True)
            for position, word in enumerate(segment.words)
        ]
        if word_rows:
            await self._session.execute(insert(TranscriptWordRow), word_rows)
