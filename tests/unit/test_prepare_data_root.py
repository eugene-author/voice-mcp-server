"""Runtime data directories per role: the api role gets no models directory, so a read-only container can start."""

import os
from pathlib import Path

import pytest

from app.core.settings import Settings, load_settings
from app.main import prepare_data_root

SHORT_VARIABLES = ("SERVICE_ROLE", "MODELS_ROOT", "DATA_ROOT", "WORKSPACE_ROOT", "ASSETS_ROOT", "DATABASE_URL")


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test without the developer's .env and short variables."""
    monkeypatch.chdir(tmp_path)
    for name in SHORT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _settings(data_root: Path) -> Settings:
    """Load settings for stdio with the data root under the test's directory."""
    return load_settings(server={"transport": "stdio"}, storage={"data_root": data_root})


def test_api_role_creates_its_directories_but_no_models_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Create the workspace and logs directories, and leave the models directory alone."""
    monkeypatch.setenv("SERVICE_ROLE", "api")
    settings = _settings(tmp_path / "data")
    prepare_data_root(settings)
    assert settings.workspace_root.is_dir()
    assert (settings.data_root / "logs").is_dir()
    assert not settings.models_root.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_api_role_starts_with_the_models_directory_on_a_read_only_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Start like the api container, whose models path sits on a read-only root filesystem."""
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    monkeypatch.setenv("SERVICE_ROLE", "api")
    monkeypatch.setenv("MODELS_ROOT", str(read_only / "models"))
    settings = _settings(tmp_path / "data")
    read_only.chmod(0o500)
    try:
        prepare_data_root(settings)
    finally:
        read_only.chmod(0o700)
    assert not settings.models_root.exists()


@pytest.mark.parametrize("role", ["all", "worker"])
def test_roles_that_run_models_create_the_models_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str
) -> None:
    """Create the models directory for every role that loads weights."""
    monkeypatch.setenv("SERVICE_ROLE", role)
    settings = _settings(tmp_path / "data")
    prepare_data_root(settings)
    assert settings.models_root.is_dir()
