"""AssetStore contract shared by every implementation."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from app.adapters.storage.local.store import LocalAssetStore
from app.adapters.storage.s3.store import S3AssetStore
from app.core.settings import S3Settings
from app.domain.assets import (
    AssetLocation,
    StorageBackend,
    job_input_key,
    job_output_key,
    voice_reference_key,
)
from app.domain.caller import Caller
from app.domain.errors import InvalidRequestError
from app.domain.ids import ClaimToken, JobId, VoiceId, uuid7
from app.ports.storage import AssetStore
from tests.mocks.storage import MockAssetStore

OWNER = Caller("https://issuer", "alice")


@pytest.fixture(params=["mock", "local", pytest.param("s3", marks=pytest.mark.s3)])
def store(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AssetStore]:
    """Return each store implementation; S3 runs against moto."""
    if request.param == "mock":
        yield MockAssetStore()
    elif request.param == "local":
        yield LocalAssetStore(tmp_path / "assets")
    else:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="contract")
            yield S3AssetStore.from_settings(S3Settings(bucket="contract", region="us-east-1"))


async def test_put_fetch_exists_delete(store: AssetStore, tmp_path: Path) -> None:
    """Round-trip bytes and report digest, size and existence."""
    source = tmp_path / "file.txt"
    source.write_text("hello")
    key = job_output_key(OWNER, JobId(uuid7()), "transcript.txt")
    stored = await store.put(source, key=key, content_type="text/plain")
    assert stored.size_bytes == 5
    assert len(stored.sha256) == 64
    assert stored.location.uri.endswith(key.value)
    assert await store.exists(stored.location)
    fetched = await store.fetch(stored.location, tmp_path / "copy.txt")
    assert fetched.read_text() == "hello"
    await store.delete(stored.location)
    await store.delete(stored.location)
    assert not await store.exists(stored.location)
    await store.verify()


async def test_put_is_idempotent_for_the_same_key(store: AssetStore, tmp_path: Path) -> None:
    """Overwrite the same key on retry without creating a second location."""
    source = tmp_path / "file.txt"
    source.write_text("v1")
    key = job_output_key(OWNER, JobId(uuid7()), "a.txt")
    first = await store.put(source, key=key, content_type="text/plain")
    source.write_text("v2")
    second = await store.put(source, key=key, content_type="text/plain")
    assert first.location == second.location
    assert first.sha256 != second.sha256


async def test_stale_objects_lists_assets_stored_under_the_key_layout(store: AssetStore, tmp_path: Path) -> None:
    """Find every kind of stored asset in the orphan scan, so the scan and the key layout agree."""
    source = tmp_path / "file.wav"
    source.write_text("audio")
    job_id = JobId(uuid7())
    keys = [
        job_output_key(OWNER, job_id, "speech.wav"),
        job_input_key(OWNER, job_id, "wav"),
        voice_reference_key(OWNER, VoiceId("narrator"), 1, "wav", claim_token=ClaimToken(uuid7())),
    ]
    stored = [await store.put(source, key=key, content_type="audio/wav") for key in keys]
    stale = await store.stale_objects(older_than=datetime.now(tz=UTC) + timedelta(minutes=5))
    assert {asset.location for asset in stored} <= set(stale)


async def test_local_store_rejects_locations_outside_root(tmp_path: Path) -> None:
    """Refuse to touch paths outside the assets root or of another backend."""
    store = LocalAssetStore(tmp_path / "assets")
    with pytest.raises(InvalidRequestError):
        await store.exists(AssetLocation(StorageBackend.LOCAL, "/etc/passwd"))
    with pytest.raises(InvalidRequestError):
        await store.exists(AssetLocation(StorageBackend.S3, "s3://b/k"))
