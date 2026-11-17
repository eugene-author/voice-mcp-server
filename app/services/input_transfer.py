"""storage.input_transfer=ingest: upload inputs to the asset store at submission, and fetch them on the worker."""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path

from app.domain.assets import INPUT_LOCATION_FIELD, UploadedInput, job_input_key
from app.domain.errors import InvalidRequestError
from app.domain.jobs import JobType, NewJob
from app.ports.storage import AssetStore
from app.services.job_planning import JobPlan
from app.services.publishing import AssetPublisher, content_type_for

logger = logging.getLogger(__name__)
UPLOAD_CONCURRENCY = 4
INPUT_PATH_FIELDS = {JobType.SPEECH_TO_TEXT: "input_path", JobType.REGISTER_VOICE: "reference_audio_path"}
"""Payload field naming the local input file of each job type that reads one."""


class InputUploader:
    """Uploads every input file of a plan and records its location in the job payload, before anything persists."""

    def __init__(self, *, store: AssetStore, max_batch_bytes: int) -> None:
        """Store the asset store and the total size limit of one submission."""
        self._store = store
        self._max_batch_bytes = max_batch_bytes

    async def upload(self, plan: JobPlan) -> JobPlan:
        """Return the plan with input_location set on each input job; raise invalid_request over the size limit."""
        targets = [
            (index, job, Path(job.input_payload[field]))
            for index, job in enumerate(plan.jobs)
            if (field := INPUT_PATH_FIELDS.get(job.type)) is not None
        ]
        if not targets:
            return plan
        sizes = await asyncio.gather(*(asyncio.to_thread(_size, path) for _, _, path in targets))
        if sum(sizes) > self._max_batch_bytes:
            raise InvalidRequestError(
                f"inputs total {sum(sizes)} bytes, over audio.max_batch_upload_bytes ({self._max_batch_bytes})"
            )
        limit = asyncio.Semaphore(UPLOAD_CONCURRENCY)
        async with asyncio.TaskGroup() as group:
            uploads = [group.create_task(self._upload(job, path, limit)) for _, job, path in targets]
        jobs = list(plan.jobs)
        for (index, job, _), upload in zip(targets, uploads, strict=True):
            payload = {**job.input_payload, INPUT_LOCATION_FIELD: upload.result().to_payload()}
            jobs[index] = replace(job, input_payload=payload)
        logger.info("uploaded inputs", extra={"files": len(targets), "bytes": sum(sizes)})
        return replace(plan, jobs=tuple(jobs))

    async def _upload(self, job: NewJob, path: Path, limit: asyncio.Semaphore) -> UploadedInput:
        """Upload one input under the job's original-input key."""
        key = job_input_key(job.owner, job.id, path.suffix.lstrip(".") or "bin")
        content_type = content_type_for(path)
        async with limit:
            stored = await self._store.put(path, key=key, content_type=content_type)
        return UploadedInput(key=key, stored=stored, content_type=content_type)


def _size(path: Path) -> int:
    """Return a file's size in bytes."""
    return path.stat().st_size


async def fetch_input(publisher: AssetPublisher, uploaded: UploadedInput | None, path: str, work: Path) -> Path:
    """Return a local copy of the job's input: fetched to the workspace when uploaded, else the shared path itself."""
    if uploaded is None:
        return Path(path)
    return await publisher.fetch(uploaded.stored.location, work / f"source{uploaded.suffix}")
