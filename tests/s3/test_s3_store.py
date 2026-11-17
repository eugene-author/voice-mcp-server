"""S3AssetStore against moto: upload with our own digest, encryption, presigned URLs and failure modes."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from app.adapters.storage.s3.store import S3AssetStore
from app.core.settings import S3Settings
from app.domain.assets import AssetLocation, StorageBackend, job_output_key
from app.domain.caller import Caller
from app.domain.errors import ConfigurationError, InvalidRequestError
from app.domain.ids import JobId, uuid7

pytestmark = pytest.mark.s3
OWNER = Caller("https://issuer", "alice")
BUCKET = "voice-private"


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[S3AssetStore]:
    """Return a store on a moto bucket with dummy credentials."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        boto3.client("s3", region_name="eu-west-1").create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
        )
        yield S3AssetStore.from_settings(S3Settings(bucket=BUCKET, region="eu-west-1", prefix="voice-mcp"))


async def test_put_fetch_delete_and_metadata(store: S3AssetStore, tmp_path: Path) -> None:
    """Upload under prefix/key with SSE and our SHA-256, then fetch and delete."""
    source = tmp_path / "t.txt"
    source.write_text("hello s3")
    key = job_output_key(OWNER, JobId(uuid7()), "transcript.txt")
    stored = await store.put(source, key=key, content_type="text/plain")
    assert stored.location.uri == f"s3://{BUCKET}/voice-mcp/{key.value}"
    head = boto3.client("s3", region_name="eu-west-1").head_object(Bucket=BUCKET, Key=f"voice-mcp/{key.value}")
    assert head["ServerSideEncryption"] == "AES256"
    assert head["Metadata"]["sha256"] == stored.sha256
    assert (await store.fetch(stored.location, tmp_path / "copy.txt")).read_text() == "hello s3"
    url = await store.presigned_url(stored.location, ttl_seconds=300)
    assert url is not None
    assert "X-Amz-Expires=300" in url
    await store.delete(stored.location)
    assert not await store.exists(stored.location)
    await store.verify()


async def test_stale_objects_listing(store: S3AssetStore, tmp_path: Path) -> None:
    """List objects under the users prefix older than a cutoff."""
    source = tmp_path / "a.txt"
    source.write_text("x")
    stored = await store.put(source, key=job_output_key(OWNER, JobId(uuid7()), "a.txt"), content_type="text/plain")
    future = datetime.now(tz=UTC) + timedelta(minutes=5)
    assert await store.stale_objects(older_than=future) == [stored.location]
    assert await store.stale_objects(older_than=future - timedelta(hours=1)) == []


async def test_missing_bucket_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail verification when the bucket does not exist."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        store = S3AssetStore.from_settings(S3Settings(bucket="missing-bucket", region="eu-west-1"))
        with pytest.raises(ConfigurationError):
            await store.verify()


async def test_rejects_foreign_locations(store: S3AssetStore) -> None:
    """Refuse local paths and malformed URIs."""
    with pytest.raises(InvalidRequestError):
        await store.exists(AssetLocation(StorageBackend.LOCAL, "/etc/passwd"))
    with pytest.raises(InvalidRequestError):
        await store.exists(AssetLocation(StorageBackend.S3, "s3://bucket-only"))
