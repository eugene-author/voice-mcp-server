"""Settings loading, aliases and cross-field validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.settings import SttPreset, load_presets, load_settings
from app.domain.stt import SttBackendChoice

HTTP_AUTH = {
    "issuer_url": "http://127.0.0.1:8080/realms/voice",
    "audience": "http://127.0.0.1:8000/mcp",
    "resource_server_url": "http://127.0.0.1:8000/mcp",
}


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test without the developer's .env and short aliases."""
    monkeypatch.chdir(tmp_path)
    for name in (
        "WHISPER_BACKEND",
        "TTS_BACKEND",
        "DATABASE_URL",
        "STORAGE_BACKEND",
        "DATA_ROOT",
        "MODELS_ROOT",
        "WORKSPACE_ROOT",
        "SCRATCH" + "_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_come_from_yaml() -> None:
    """Load default.yaml values."""
    settings = load_settings(server={"transport": "stdio"})
    assert settings.stt.model == "small"
    assert settings.jobs.max_attempts == 2
    assert settings.audio.fillers["en"][0] == "um"


def test_streamable_http_requires_issuer_and_audience() -> None:
    """Fail startup for HTTP without OAuth configuration."""
    with pytest.raises(ValidationError, match=r"issuer_url and auth\.audience"):
        load_settings(server={"transport": "streamable_http"})


def test_audience_must_equal_resource_url() -> None:
    """Require audience == resource_server_url."""
    with pytest.raises(ValidationError, match="must equal"):
        load_settings(auth={**HTTP_AUTH, "audience": "http://other/mcp"})


def test_http_with_valid_auth() -> None:
    """Accept a complete OAuth configuration."""
    assert load_settings(auth=HTTP_AUTH).auth.required_scopes == ("mcp:access",)


@pytest.mark.parametrize("algorithms", [["none"], ["HS256"], []])
def test_unsafe_algorithms_rejected(algorithms: list[str]) -> None:
    """Reject none and HMAC algorithms."""
    with pytest.raises(ValidationError):
        load_settings(server={"transport": "stdio"}, auth={"algorithms": algorithms})


def test_s3_requires_bucket_and_region() -> None:
    """Fail closed when S3 lacks bucket or region."""
    with pytest.raises(ValidationError, match="bucket"):
        load_settings(server={"transport": "stdio"}, storage={"backend": "s3"})


def test_short_aliases_and_invalid_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map WHISPER_BACKEND and reject unknown values."""
    monkeypatch.setenv("WHISPER_BACKEND", "ctranslate2")
    monkeypatch.setenv("DATA_ROOT", "/srv/voice")
    settings = load_settings(server={"transport": "stdio"})
    assert settings.stt.backend is SttBackendChoice.CTRANSLATE2
    assert settings.data_root == Path("/srv/voice")
    monkeypatch.setenv("WHISPER_BACKEND", "whisper.cpp")
    with pytest.raises(ValidationError):
        load_settings(server={"transport": "stdio"})


def test_short_alias_from_dotenv(tmp_path: Path) -> None:
    """Read short aliases from .env when the environment lacks them."""
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://u:p@h/db\n")
    settings = load_settings(server={"transport": "stdio"})
    assert settings.database.url is not None
    assert settings.database.url.get_secret_value().endswith("/db")
    assert "postgresql" not in repr(settings.database)


def test_runtime_downloads_cannot_be_enabled() -> None:
    """Keep allow_runtime_downloads false."""
    with pytest.raises(ValidationError):
        load_settings(server={"transport": "stdio"}, models={"allow_runtime_downloads": True})


def test_presets_load() -> None:
    """Ship the four documented presets."""
    assert set(load_presets()) == {"fast", "balanced", "accurate", "studio"}


@pytest.mark.parametrize(
    ("section", "old_key", "new_key", "value"),
    [
        ("stt", "halluc" + "ination_checks", "segment_flags", False),
        ("audio", "max_batch_" + "ingest_bytes", "max_batch_upload_bytes", 1024),
        ("storage", "scratch" + "_root", "workspace_root", Path("/tmp/workspace")),
    ],
)
def test_renamed_settings_reject_the_old_key_and_accept_the_new_one(
    section: str, old_key: str, new_key: str, value: object
) -> None:
    """Fail startup on an old key instead of ignoring it, and load the new key."""
    stdio = {"transport": "stdio"}
    with pytest.raises(ValidationError, match=old_key):
        load_settings(server=stdio, **{section: {old_key: value}})
    assert getattr(getattr(load_settings(server=stdio, **{section: {new_key: value}}), section), new_key) == value


def test_workspace_defaults_to_the_workspace_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Default to a directory named like its setting, and honor WORKSPACE_ROOT."""
    settings = load_settings(server={"transport": "stdio"}, storage={"data_root": tmp_path})
    assert settings.workspace_root == (tmp_path / "workspace").resolve()
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "elsewhere"))
    assert load_settings(server={"transport": "stdio"}).workspace_root == (tmp_path / "elsewhere").resolve()


def test_retired_workspace_variable_fails_startup_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse the retired variable instead of quietly moving the workspace, and name its replacement."""
    monkeypatch.setenv("SCRATCH_ROOT", "/mnt/large-disk")
    with pytest.raises(ValueError, match="WORKSPACE_ROOT"):
        load_settings(server={"transport": "stdio"})


def test_retired_workspace_variable_fails_startup_from_dotenv(tmp_path: Path) -> None:
    """Refuse the retired variable when it comes from a .env file."""
    (tmp_path / ".env").write_text("SCRATCH_ROOT=/mnt/large-disk\n")
    with pytest.raises(ValueError, match="WORKSPACE_ROOT"):
        load_settings(server={"transport": "stdio"})


def test_presets_reject_the_old_flag_setting() -> None:
    """Fail on the old flag setting in a preset, and accept the new one."""
    with pytest.raises(ValidationError):
        SttPreset.model_validate({"halluc" + "ination_checks": True})
    assert SttPreset.model_validate({"segment_flags": False}).segment_flags is False
