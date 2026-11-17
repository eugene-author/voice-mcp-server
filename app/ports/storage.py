"""Asset storage port; it handles bytes while AssetRepository records metadata rows."""

from datetime import datetime
from pathlib import Path
from typing import Protocol

from app.domain.assets import AssetKey, AssetLocation, StorageBackend, StoredAsset


class AssetStore(Protocol):
    """Saves and loads asset bytes in the configured backend."""

    @property
    def backend(self) -> StorageBackend:
        """Return the storage backend identifier."""
        ...

    async def verify(self) -> None:
        """Raise when the backend is not reachable or not writable."""
        ...

    async def put(self, source: Path, *, key: AssetKey, content_type: str) -> StoredAsset:
        """Publish a finalized local file and return its durable location, digest and size."""
        ...

    async def fetch(self, location: AssetLocation, destination: Path) -> Path:
        """Copy a stored asset to a local path."""
        ...

    async def exists(self, location: AssetLocation) -> bool:
        """Return whether the asset exists."""
        ...

    async def delete(self, location: AssetLocation) -> None:
        """Delete the asset; deleting a missing asset is not an error."""
        ...

    async def presigned_url(self, location: AssetLocation, *, ttl_seconds: int) -> str | None:
        """Return a short-lived download URL, or None when the backend has no such URLs."""
        ...

    async def stale_objects(self, *, older_than: datetime) -> list[AssetLocation]:
        """Return stored assets last modified before older_than, for orphan reconciliation."""
        ...
