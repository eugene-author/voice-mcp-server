"""Creating the configured database before migrations, against a real PostgreSQL server."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest
from click.testing import Result
from pydantic import SecretStr
from sqlalchemy.engine import make_url
from typer.testing import CliRunner

from app.cli import app
from app.core.settings import DatabaseSettings
from app.db.provisioning import ensure_database
from app.domain.errors import ConfigurationError
from tests.integration.conftest import TEST_DATABASE_URL

pytestmark = pytest.mark.integration

SERVER = make_url(TEST_DATABASE_URL)
MAINTENANCE_URL = SERVER.set(database="postgres").render_as_string(hide_password=False)
SECRET_PASSWORD = "not-the-password"


def _url(*, database: str, username: str | None = None, password: str | None = None) -> str:
    """Return a URL for the test server with the given database and, optionally, another login."""
    url = SERVER.set(database=database)
    if username is not None:
        url = url.set(username=username, password=password)
    return url.render_as_string(hide_password=False)


def _settings(url: str) -> DatabaseSettings:
    """Return database settings for a URL."""
    return DatabaseSettings(url=SecretStr(url))


async def _exists(name: str) -> bool:
    """Return whether the server has a database with this name."""
    connection = await asyncpg.connect(MAINTENANCE_URL)
    try:
        return bool(await connection.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name))
    finally:
        await connection.close()


@pytest.fixture
async def database_name() -> AsyncIterator[str]:
    """Yield an unused database name that needs quoting, and drop the database afterwards."""
    name = f"voice-provisioning-{uuid.uuid4().hex[:8]}"
    yield name
    connection = await asyncpg.connect(MAINTENANCE_URL)
    try:
        await connection.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await connection.close()


async def _invoke(runner: CliRunner, arguments: list[str]) -> Result:
    """Run the CLI on a worker thread, because migrate starts its own event loop."""
    return await asyncio.to_thread(runner.invoke, app, arguments)


async def test_creates_a_missing_database_once(database_name: str) -> None:
    """Create the database on the first call and leave it alone on the second."""
    settings = _settings(_url(database=database_name))
    assert await ensure_database(settings, create=True) is True
    assert await _exists(database_name)
    assert await ensure_database(settings, create=True) is False


async def test_missing_database_is_reported_without_creating_it(database_name: str) -> None:
    """Name the missing database and the flag that creates it, and leave the server unchanged."""
    with pytest.raises(ConfigurationError) as caught:
        await ensure_database(_settings(_url(database=database_name)), create=False)
    assert database_name in caught.value.message
    assert "--create-database" in caught.value.message
    assert not await _exists(database_name)


async def test_existing_database_needs_no_creation() -> None:
    """Report nothing to do for a database that is already there, whether or not creation is allowed."""
    assert await ensure_database(_settings(TEST_DATABASE_URL), create=False) is False
    assert await ensure_database(_settings(TEST_DATABASE_URL), create=True) is False


async def test_wrong_password_names_the_server_and_hides_the_password() -> None:
    """Explain a rejected login in one line, without repeating the password."""
    url = _url(database="postgres", username=str(SERVER.username), password=SECRET_PASSWORD)
    with pytest.raises(ConfigurationError) as caught:
        await ensure_database(_settings(url), create=False)
    message = caught.value.message
    assert f"{SERVER.host}:{SERVER.port}" in message
    assert "DATABASE_URL" in message
    assert SECRET_PASSWORD not in message
    assert "\n" not in message


async def test_login_without_create_permission_is_explained(database_name: str) -> None:
    """Say the database must be created by hand when the login may not create one."""
    role = f"voice_provisioning_{uuid.uuid4().hex[:8]}"
    connection = await asyncpg.connect(MAINTENANCE_URL)
    try:
        await connection.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'role-password' NOCREATEDB")
        with pytest.raises(ConfigurationError, match="cannot create"):
            await ensure_database(
                _settings(_url(database=database_name, username=role, password="role-password")), create=True
            )
    finally:
        await connection.execute(f"DROP ROLE {role}")
        await connection.close()
    assert not await _exists(database_name)


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CliRunner:
    """Return a CLI runner outside the repository, so its .env is not read."""
    monkeypatch.chdir(tmp_path)
    return CliRunner()


async def test_migrate_creates_the_database_and_migrates_it(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, database_name: str
) -> None:
    """Take a server with no database for this install all the way to the latest schema."""
    monkeypatch.setenv("DATABASE_URL", _url(database=database_name))
    result = await _invoke(runner, ["migrate", "--create-database"])
    assert result.exit_code == 0, result.output
    assert "created" in result.output
    connection = await asyncpg.connect(_url(database=database_name))
    try:
        assert await connection.fetchval("SELECT count(*) FROM alembic_version") == 1
    finally:
        await connection.close()
    second = await _invoke(runner, ["migrate", "--create-database"])
    assert second.exit_code == 0, second.output
    assert "created" not in second.output


async def test_migrate_without_the_flag_reports_a_missing_database(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, database_name: str
) -> None:
    """Exit with a short message that names --create-database, and create nothing."""
    monkeypatch.setenv("DATABASE_URL", _url(database=database_name))
    result = await _invoke(runner, ["migrate"])
    assert result.exit_code == 1
    assert "--create-database" in result.output
    assert "Traceback" not in result.output
    assert not await _exists(database_name)


async def test_migrate_reports_a_rejected_login_in_one_short_message(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replace the long traceback with a message that points at DATABASE_URL."""
    monkeypatch.setenv("DATABASE_URL", _url(database="voice", username=str(SERVER.username), password=SECRET_PASSWORD))
    result = await _invoke(runner, ["migrate"])
    assert result.exit_code == 1
    assert "DATABASE_URL" in result.output
    assert "Traceback" not in result.output
    assert SECRET_PASSWORD not in result.output
    assert len(result.output.splitlines()) <= 3
