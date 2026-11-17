"""Connection failures before migrations are explained in one line and need no database server."""

from pathlib import Path

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from app.cli import app
from app.core.settings import DatabaseSettings
from app.db.provisioning import ensure_database
from app.domain.errors import ConfigurationError

UNREACHABLE = "postgresql://voice:unreachable-secret@127.0.0.1:1/voice"


async def test_unreachable_server_names_the_address_and_hides_the_password() -> None:
    """Explain that nothing is listening, without repeating the password."""
    settings = DatabaseSettings(url=SecretStr(UNREACHABLE), connect_timeout_seconds=2)
    with pytest.raises(ConfigurationError) as caught:
        await ensure_database(settings, create=False)
    assert "127.0.0.1:1" in caught.value.message
    assert "unreachable-secret" not in caught.value.message
    assert "\n" not in caught.value.message


async def test_missing_url_is_explained() -> None:
    """Point at DATABASE_URL when it is not set."""
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        await ensure_database(DatabaseSettings(), create=False)


def test_migrate_fails_with_a_short_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exit non-zero with one short message and no traceback when the server is unreachable."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", UNREACHABLE)
    result = CliRunner().invoke(app, ["migrate"])
    assert result.exit_code == 1
    assert "127.0.0.1:1" in result.output
    assert "Traceback" not in result.output
    assert len(result.output.splitlines()) <= 3
