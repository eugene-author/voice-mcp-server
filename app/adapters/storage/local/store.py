"""Local filesystem asset store with atomic publication."""

import asyncio
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from app.adapters.storage.checksums import sha256_file
from app.domain.assets import USERS_ROOT, AssetKey, AssetLocation, StorageBackend, StoredAsset
from app.domain.errors import AssetPublishError, ConfigurationError, InvalidRequestError


class LocalAssetStore:
    """Stores assets under an assets root using the owner-scoped key layout."""

    def __init__(self, root: Path) -> None:
        """Create the assets root with restrictive permissions."""
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True, mode=0o750)

    @property
    def backend(self) -> StorageBackend:
        """Return the storage backend identifier."""
        return StorageBackend.LOCAL

    async def verify(self) -> None:
        """Raise when the assets root is not a writable directory."""
        if not self._root.is_dir() or not os.access(self._root, os.W_OK):
            raise ConfigurationError("the assets root is not a writable directory")

    async def put(self, source: Path, *, key: AssetKey, content_type: str) -> StoredAsset:
        """Copy the finalized file next to its destination, fsync it, then rename it into place atomically."""
        try:
            return await asyncio.to_thread(self._put_sync, source, key)
        except OSError as exc:
            raise AssetPublishError(f"could not publish asset {key.value.rsplit('/', 1)[-1]}") from exc

    async def fetch(self, location: AssetLocation, destination: Path) -> Path:
        """Copy a stored asset to a local path."""
        path = self._path_of(location)
        destination.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, path, destination)
        return destination

    async def exists(self, location: AssetLocation) -> bool:
        """Return whether the asset exists."""
        return self._path_of(location).is_file()

    async def delete(self, location: AssetLocation) -> None:
        """Delete the asset; deleting a missing asset is not an error."""
        self._path_of(location).unlink(missing_ok=True)

    async def presigned_url(self, location: AssetLocation, *, ttl_seconds: int) -> str | None:
        """Return None: local assets are addressed by path."""
        return None

    async def stale_objects(self, *, older_than: datetime) -> list[AssetLocation]:
        """Return finalized files under users/ last modified before older_than."""
        return await asyncio.to_thread(self._stale_sync, older_than.timestamp())

    def _stale_sync(self, cutoff: float) -> list[AssetLocation]:
        """Walk the users tree on a worker thread, skipping in-progress staging files."""
        users = self._root / USERS_ROOT
        if not users.is_dir():
            return []
        return [
            AssetLocation(backend=StorageBackend.LOCAL, uri=str(path))
            for path in users.rglob("*")
            if path.is_file() and not path.name.endswith(".part") and path.stat().st_mtime < cutoff
        ]

    def _put_sync(self, source: Path, key: AssetKey) -> StoredAsset:
        """Publish synchronously; runs on a worker thread."""
        destination = self._root / key.value
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            shutil.copyfile(source, staging)
            with staging.open("rb") as handle:
                os.fsync(handle.fileno())
            digest = sha256_file(staging)
            size = staging.stat().st_size
            os.replace(staging, destination)
        finally:
            staging.unlink(missing_ok=True)
        location = AssetLocation(backend=StorageBackend.LOCAL, uri=str(destination))
        return StoredAsset(location=location, sha256=digest, size_bytes=size)

    def _path_of(self, location: AssetLocation) -> Path:
        """Return the local path of a location, rejecting other backends and paths outside the root."""
        if location.backend is not StorageBackend.LOCAL:
            raise InvalidRequestError("asset is not stored in the local backend")
        path = Path(location.uri).resolve()
        if not path.is_relative_to(self._root):
            raise InvalidRequestError("asset location is outside the assets root")
        return path
