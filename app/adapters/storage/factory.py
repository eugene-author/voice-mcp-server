"""Builds the configured asset store; S3 is imported lazily so boto3 is only needed in S3 mode."""

from pathlib import Path

from app.adapters.storage.local.store import LocalAssetStore
from app.core.settings import Settings
from app.ports.storage import AssetStore


def build_asset_store(settings: Settings) -> AssetStore:
    """Return the store for storage.backend, without falling back from S3 to local."""
    if settings.storage.backend == "s3":
        from app.adapters.storage.s3.store import S3AssetStore

        return S3AssetStore.from_settings(settings.storage.s3)
    return LocalAssetStore(Path(settings.assets_root))
