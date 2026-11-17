"""In-memory asset store."""

import asyncio
import hashlib
from datetime import datetime
from pathlib import Path

from app.domain.assets import AssetKey, AssetLocation, StorageBackend, StoredAsset
from app.domain.errors import AssetPublishError, ConfigurationError, NotFoundError


class MockAssetStore:
    """Keeps asset bytes in a dictionary keyed by URI."""

    def __init__(self, *, backend: StorageBackend = StorageBackend.LOCAL, fail_puts: int = 0) -> None:
        """Configure the reported backend and how many puts fail before succeeding."""
        self._backend = backend
        self.fail_puts = fail_puts
        self.fail_deletes_after: int | None = None
        """When set, deletes succeed this many more times and then raise until reset to None."""
        self.blobs: dict[str, bytes] = {}
        self.modified_at: dict[str, datetime] = {}
        self.healthy = True

    @property
    def backend(self) -> StorageBackend:
        """Return the configured backend."""
        return self._backend

    async def verify(self) -> None:
        """Raise when marked unhealthy."""
        if not self.healthy:
            raise ConfigurationError("store unavailable")

    async def put(self, source: Path, *, key: AssetKey, content_type: str) -> StoredAsset:
        """Store bytes under a backend-shaped URI."""
        if self.fail_puts > 0:
            self.fail_puts -= 1
            raise AssetPublishError("simulated upload failure")
        data = await asyncio.to_thread(source.read_bytes)
        uri = f"s3://bucket/{key.value}" if self._backend is StorageBackend.S3 else f"/assets/{key.value}"
        self.blobs[uri] = data
        self.modified_at.setdefault(uri, datetime.min)
        location = AssetLocation(self._backend, uri)
        return StoredAsset(location=location, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))

    async def fetch(self, location: AssetLocation, destination: Path) -> Path:
        """Write stored bytes to destination."""
        if location.uri not in self.blobs:
            raise NotFoundError("asset not found")
        await asyncio.to_thread(destination.write_bytes, self.blobs[location.uri])
        return destination

    async def exists(self, location: AssetLocation) -> bool:
        """Return whether bytes are stored."""
        return location.uri in self.blobs

    async def delete(self, location: AssetLocation) -> None:
        """Remove bytes if present, or fail as configured by fail_deletes_after."""
        if self.fail_deletes_after is not None:
            if self.fail_deletes_after <= 0:
                raise AssetPublishError("simulated delete failure")
            self.fail_deletes_after -= 1
        self.blobs.pop(location.uri, None)

    async def presigned_url(self, location: AssetLocation, *, ttl_seconds: int) -> str | None:
        """Return a placeholder presigned URL in S3 mode."""
        if self._backend is not StorageBackend.S3:
            return None
        return f"https://example.invalid/{location.uri.removeprefix('s3://')}?ttl={ttl_seconds}"

    async def stale_objects(self, *, older_than: datetime) -> list[AssetLocation]:
        """Return stored assets whose recorded modification time is before older_than."""
        cutoff = older_than.replace(tzinfo=None)
        return [
            AssetLocation(self._backend, uri)
            for uri, at in self.modified_at.items()
            if uri in self.blobs and at < cutoff
        ]
