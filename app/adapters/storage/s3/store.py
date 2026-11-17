"""Amazon S3 asset store; boto3 calls run on worker threads so they do not block the event loop."""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.adapters.storage.checksums import sha256_file
from app.core.settings import S3Settings
from app.domain.assets import USERS_ROOT, AssetKey, AssetLocation, StorageBackend, StoredAsset
from app.domain.errors import AssetPublishError, ConfigurationError, InvalidRequestError, MediaError

_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


@dataclass(frozen=True, slots=True)
class S3Object:
    """Object listed under the configured prefix."""

    key: str
    size_bytes: int
    last_modified_epoch: float


def _digest_and_size(path: Path) -> tuple[str, int]:
    """Return a file's SHA-256 and size; runs on a worker thread."""
    return sha256_file(path), path.stat().st_size


class S3AssetStore:
    """Stores assets in a private S3 bucket under a fixed prefix."""

    def __init__(
        self, *, client: Any, bucket: str, prefix: str, encryption: str, kms_key_id: str | None, max_concurrency: int
    ) -> None:
        """Store the boto3 client and bucket layout."""
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._encryption = encryption
        self._kms_key_id = kms_key_id
        self._transfer = TransferConfig(max_concurrency=max_concurrency, use_threads=True)

    @classmethod
    def from_settings(cls, settings: S3Settings) -> S3AssetStore:
        """Create a client from the default boto3 credential chain with bounded retries and timeouts."""
        if not settings.bucket or not settings.region:
            raise ConfigurationError("storage.backend=s3 requires bucket and region")
        config = Config(
            region_name=settings.region,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
        )
        return cls(
            client=boto3.client("s3", config=config),
            bucket=settings.bucket,
            prefix=settings.prefix,
            encryption=settings.encryption,
            kms_key_id=settings.kms_key_id,
            max_concurrency=settings.max_concurrency,
        )

    @property
    def backend(self) -> StorageBackend:
        """Return the storage backend identifier."""
        return StorageBackend.S3

    @property
    def bucket(self) -> str:
        """Return the bucket name."""
        return self._bucket

    async def verify(self) -> None:
        """Fail closed when the bucket is missing or not accessible with the current credentials."""
        try:
            await asyncio.to_thread(self._client.head_bucket, Bucket=self._bucket)
        except (BotoCoreError, ClientError) as exc:
            raise ConfigurationError(f"S3 bucket {self._bucket!r} is not accessible") from exc

    async def put(self, source: Path, *, key: AssetKey, content_type: str) -> StoredAsset:
        """Upload a finalized file, record our own SHA-256 and verify the stored size."""
        object_key = self.object_key(key)
        try:
            digest, size = await asyncio.to_thread(_digest_and_size, source)
            extra: dict[str, Any] = {"ContentType": content_type, "Metadata": {"sha256": digest}}
            if self._encryption == "sse-kms":
                extra["ServerSideEncryption"] = "aws:kms"
                if self._kms_key_id:
                    extra["SSEKMSKeyId"] = self._kms_key_id
            else:
                extra["ServerSideEncryption"] = "AES256"
            await asyncio.to_thread(
                self._client.upload_file, str(source), self._bucket, object_key, ExtraArgs=extra, Config=self._transfer
            )
            head = await asyncio.to_thread(self._client.head_object, Bucket=self._bucket, Key=object_key)
        except (BotoCoreError, ClientError, OSError) as exc:
            raise AssetPublishError(f"could not upload asset {key.value.rsplit('/', 1)[-1]}") from exc
        if int(head.get("ContentLength", -1)) != size:
            raise AssetPublishError("uploaded object size does not match the staged file")
        location = AssetLocation(backend=StorageBackend.S3, uri=f"s3://{self._bucket}/{object_key}")
        return StoredAsset(location=location, sha256=digest, size_bytes=size)

    async def fetch(self, location: AssetLocation, destination: Path) -> Path:
        """Download an object to a local path."""
        bucket, object_key = self._parse(location)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(self._client.download_file, bucket, object_key, str(destination))
        except (BotoCoreError, ClientError) as exc:
            raise MediaError("could not download the stored asset") from exc
        return destination

    async def exists(self, location: AssetLocation) -> bool:
        """Return whether the object exists."""
        bucket, object_key = self._parse(location)
        try:
            await asyncio.to_thread(self._client.head_object, Bucket=bucket, Key=object_key)
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code")) in _NOT_FOUND_CODES:
                return False
            raise AssetPublishError("could not check the stored asset") from exc
        return True

    async def delete(self, location: AssetLocation) -> None:
        """Delete the object; deleting a missing object is not an error."""
        bucket, object_key = self._parse(location)
        try:
            await asyncio.to_thread(self._client.delete_object, Bucket=bucket, Key=object_key)
        except (BotoCoreError, ClientError) as exc:
            raise AssetPublishError("could not delete the stored asset") from exc

    async def presigned_url(self, location: AssetLocation, *, ttl_seconds: int) -> str | None:
        """Return a short-lived GET URL; it must not be persisted or logged."""
        bucket, object_key = self._parse(location)
        url = await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": bucket, "Key": object_key},
            ExpiresIn=ttl_seconds,
        )
        return str(url)

    async def stale_objects(self, *, older_than: datetime) -> list[AssetLocation]:
        """Return objects under prefix/users/ last modified before older_than."""
        prefix = f"{self._prefix}/{USERS_ROOT}/" if self._prefix else f"{USERS_ROOT}/"
        try:
            objects = await asyncio.to_thread(lambda: list(self._iter_objects(prefix)))
        except (BotoCoreError, ClientError) as exc:
            raise AssetPublishError("could not list stored assets") from exc
        cutoff = older_than.timestamp()
        return [self.location_of(item.key) for item in objects if item.last_modified_epoch < cutoff]

    def object_key(self, key: AssetKey) -> str:
        """Return the bucket object key of an asset key."""
        return f"{self._prefix}/{key.value}" if self._prefix else key.value

    def location_of(self, object_key: str) -> AssetLocation:
        """Return the location of an object key in this bucket."""
        return AssetLocation(backend=StorageBackend.S3, uri=f"s3://{self._bucket}/{object_key}")

    def _iter_objects(self, prefix: str) -> Iterator[S3Object]:
        """Yield listed objects page by page; runs on a worker thread."""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield S3Object(item["Key"], int(item["Size"]), item["LastModified"].timestamp())

    @staticmethod
    def _parse(location: AssetLocation) -> tuple[str, str]:
        """Split an s3:// URI into bucket and key."""
        if location.backend is not StorageBackend.S3 or not location.uri.startswith("s3://"):
            raise InvalidRequestError("asset is not stored in S3")
        bucket, _, object_key = location.uri.removeprefix("s3://").partition("/")
        if not bucket or not object_key:
            raise InvalidRequestError("invalid S3 location")
        return bucket, object_key
