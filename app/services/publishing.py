"""Publishes staged files through the AssetStore and fetches uploaded inputs; rows are recorded by JobTransitions."""

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from app.domain.assets import AssetKey, AssetLocation, AssetRole, PublishedAsset
from app.ports.storage import AssetStore

CONTENT_TYPES = {
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
    ".srt": "application/x-subrip",
    ".vtt": "text/vtt",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".opus": "audio/ogg",
    ".webm": "audio/webm",
}


def content_type_for(path: Path) -> str:
    """Return the content type of a file by extension."""
    suffix = path.suffix.lower()
    return CONTENT_TYPES.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


@dataclass(frozen=True, slots=True)
class StagedFile:
    """Finalized local file waiting to be published."""

    name: str
    path: Path
    role: AssetRole
    key: AssetKey
    duration_seconds: float | None = None
    sample_rate: int | None = None


class AssetPublisher:
    """Puts staged files into the configured store."""

    def __init__(self, store: AssetStore) -> None:
        """Store the asset store."""
        self._store = store

    async def publish(self, files: list[StagedFile]) -> tuple[PublishedAsset, ...]:
        """Publish each file and return what to record; failures raise AssetPublishError."""
        published: list[PublishedAsset] = []
        for staged in files:
            content_type = content_type_for(staged.path)
            stored = await self._store.put(staged.path, key=staged.key, content_type=content_type)
            published.append(
                PublishedAsset(
                    name=staged.name,
                    role=staged.role,
                    key=staged.key,
                    stored=stored,
                    content_type=content_type,
                    duration_seconds=staged.duration_seconds,
                    sample_rate=staged.sample_rate,
                )
            )
        return tuple(published)

    async def fetch(self, location: AssetLocation, destination: Path) -> Path:
        """Copy a stored asset, such as an uploaded input, to a local path."""
        return await self._store.fetch(location, destination)
