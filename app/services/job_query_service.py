"""Read-only job queries; they do not load models or touch the inference executor."""

from dataclasses import dataclass

from app.domain.assets import AssetLocation, StorageBackend
from app.domain.caller import Caller
from app.domain.errors import InvalidRequestError, NotFoundError
from app.domain.ids import JobId
from app.domain.jobs import JobCursor, JobFilter, JobPage, JobView
from app.ports.storage import AssetStore
from app.ports.unit_of_work import UnitOfWorkFactory


@dataclass(frozen=True, slots=True)
class JobDetails:
    """Job view plus optional short-lived download URLs."""

    view: JobView
    download_urls: dict[str, str]


class JobQueryService:
    """Owner-scoped get_job and list_jobs projections."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkFactory,
        store: AssetStore,
        default_limit: int,
        max_limit: int,
        presign_enabled: bool,
        presign_ttl_seconds: int,
    ) -> None:
        """Store injected collaborators and listing limits."""
        self._uow = uow
        self._store = store
        self._default_limit = default_limit
        self._max_limit = max_limit
        self._presign_enabled = presign_enabled
        self._presign_ttl_seconds = presign_ttl_seconds

    async def get(self, job_id: JobId, *, owner: Caller) -> JobView:
        """Return an owned job or raise not_found, identical for missing and foreign jobs."""
        async with self._uow() as uow:
            view = await uow.jobs.get(job_id, owner)
        if view is None:
            raise NotFoundError("job not found")
        return view

    async def details(self, job_id: JobId, *, owner: Caller, include_download_urls: bool) -> JobDetails:
        """Return an owned job and, on explicit request in S3 mode, presigned URLs for its files."""
        view = await self.get(job_id, owner=owner)
        if not include_download_urls:
            return JobDetails(view=view, download_urls={})
        if not self._presign_enabled or self._store.backend is not StorageBackend.S3:
            raise InvalidRequestError("download URLs are only available in S3 mode with presign_downloads enabled")
        urls: dict[str, str] = {}
        for name, uri in view.files.items():
            if uri.startswith("s3://"):
                url = await self._store.presigned_url(
                    AssetLocation(StorageBackend.S3, uri), ttl_seconds=self._presign_ttl_seconds
                )
                if url is not None:
                    urls[name] = url
        return JobDetails(view=view, download_urls=urls)

    async def list(self, *, owner: Caller, job_filter: JobFilter, limit: int | None, cursor: str | None) -> JobPage:
        """Return one page of the owner's jobs, newest first."""
        effective_limit = self._default_limit if limit is None else limit
        if not 1 <= effective_limit <= self._max_limit:
            raise InvalidRequestError(f"limit must be between 1 and {self._max_limit}")
        position = JobCursor.decode(cursor) if cursor else None
        async with self._uow() as uow:
            return await uow.jobs.page(owner, job_filter, limit=effective_limit, cursor=position)
