"""Batch parents: idempotent aggregation of child statuses and the final manifest."""

import json
import logging

from app.domain.assets import AssetRole, job_output_key
from app.domain.ids import JobId
from app.domain.jobs import JobEvent, JobStatus
from app.ports.clock import Clock
from app.ports.job_queue import JobWaiters
from app.ports.unit_of_work import UnitOfWorkFactory
from app.services.job_workspace import JobWorkspace
from app.services.publishing import AssetPublisher, StagedFile

logger = logging.getLogger(__name__)
MANIFEST_NAME = "manifest.json"


class BatchService:
    """Recomputes parent counts from child statuses, so each child counts once even across retries."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkFactory,
        publisher: AssetPublisher,
        workspace: JobWorkspace,
        waiters: JobWaiters,
        clock: Clock,
    ) -> None:
        """Store injected collaborators."""
        self._uow = uow
        self._publisher = publisher
        self._workspace = workspace
        self._waiters = waiters
        self._clock = clock

    async def aggregate(self, parent_id: JobId) -> JobStatus | None:
        """Update a parent's counts and, once every child finished, publish its manifest and finalize it."""
        async with self._uow() as uow:
            parent = await uow.jobs.get(parent_id, None)
            if parent is None or parent.status.is_terminal:
                return None
            counts = await uow.jobs.batch_counts(parent_id)
            now = self._clock.now()
            if parent.status is JobStatus.QUEUED and counts.finished > 0:
                await uow.jobs.update_batch_parent(parent_id, counts=counts, status=JobStatus.RUNNING, now=now)
            else:
                await uow.jobs.update_batch_parent(parent_id, counts=counts, status=None, now=now)
            final = counts.final_status()
            if final is None:
                return None
            owner = await uow.jobs.owner_of(parent_id)
            children = await uow.jobs.batch_children(parent_id)
        if owner is None:
            return None
        manifest_path = self._workspace.staged_dir(parent_id) / MANIFEST_NAME
        manifest = {
            "batch_job_id": str(parent_id),
            "counts": counts.to_dict(),
            "status": str(final),
            "items": [
                {
                    "job_id": str(child.job_id),
                    "input_path": child.input_path,
                    "status": str(child.status),
                    "files": child.files,
                    "error": {"code": child.error.code, "message": child.error.message} if child.error else None,
                }
                for child in children
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        assets = await self._publisher.publish(
            [
                StagedFile(
                    name="manifest",
                    path=manifest_path,
                    role=AssetRole.MANIFEST,
                    key=job_output_key(owner, parent_id, MANIFEST_NAME),
                )
            ]
        )
        async with self._uow() as uow:
            now = self._clock.now()
            for asset in assets:
                await uow.assets.add(asset, owner=owner, job_id=parent_id, voice_version_id=None, created_at=now)
            current = await uow.jobs.get(parent_id, None)
            if current is not None and current.status is JobStatus.QUEUED:
                await uow.jobs.update_batch_parent(parent_id, counts=counts, status=JobStatus.RUNNING, now=now)
            changed = await uow.jobs.update_batch_parent(parent_id, counts=counts, status=final, now=now)
            if changed:
                await uow.events.add(
                    JobEvent(job_id=parent_id, at=now, event_type="batch_finished", message=str(final))
                )
        self._workspace.remove_job(parent_id)
        self._waiters.resolve_if_terminal(parent_id, final)
        logger.info("batch finished", extra={"job_id": str(parent_id), "status": str(final)})
        return final

    async def reconcile_open_parents(self) -> None:
        """Re-aggregate every non-terminal parent, recovering from crashes between child and parent updates."""
        async with self._uow() as uow:
            parent_ids = await uow.jobs.open_batch_parent_ids()
        for parent_id in parent_ids:
            await self.aggregate(parent_id)
