"""On-demand GPU worker settings: the master switch, every validator, and the short environment aliases."""

import subprocess
import sys
import textwrap
from datetime import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.settings import Settings, load_settings
from app.domain.models import ModelKind
from app.domain.workers import ModelRef

ROOT = Path(__file__).resolve().parents[2]
ALIASES = ("SERVICE_ROLE", "GPU_ON_DEMAND", "DATABASE_URL")


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test without the developer's .env or aliases."""
    monkeypatch.chdir(tmp_path)
    for name in ALIASES:
        monkeypatch.delenv(name, raising=False)


def _settings(role: str = "api", **on_demand: Any) -> Settings:
    """Return stdio settings with a role and on_demand values."""
    return load_settings(server={"transport": "stdio"}, deployment={"role": role}, gpu_worker={"on_demand": on_demand})


def _window(**values: Any) -> dict[str, Any]:
    """Return a valid window with overrides."""
    return {"days": ["mon"], "start": "08:00", "end": "10:00", **values}


def test_defaults_keep_todays_behavior() -> None:
    """Default to one all-role process, no controller, shared input paths and no idle unload."""
    settings = load_settings(server={"transport": "stdio"})
    assert settings.deployment.role == "all"
    assert not settings.gpu_worker.on_demand.enabled
    assert settings.gpu_worker.on_demand.target == "none"
    assert settings.gpu_worker.idle_unload_seconds == 0
    assert settings.storage.input_transfer == "shared_path"


def _imports_after_building(settings_code: str) -> list[str]:
    """Build the wake controller in a fresh interpreter and return the compute, Docker and AWS modules it loaded."""
    script = textwrap.dedent(f"""
        import asyncio, sys
        from contextlib import AsyncExitStack
        import app.cli, app.main
        from app.core.settings import load_settings

        async def build():
            settings = {settings_code}
            async with AsyncExitStack() as stack:
                built = app.main.build_wake_controller(
                    settings, engine=None, uow=None, clock=None, submitted=asyncio.Event(), stack=stack
                )
                print("controller" if built else "none")

        asyncio.run(build())
        print(*sorted(m for m in sys.modules if m.startswith(("app.adapters.compute", "boto3", "botocore"))))
        """)
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.split()


def test_master_switch_off_constructs_and_imports_nothing() -> None:
    """Construct no controller and import no compute adapter, Docker or AWS client with the switch off."""
    off = 'load_settings(server={"transport": "stdio"}, deployment={"role": "api"})'
    assert _imports_after_building(off) == ["none"]


def test_master_switch_on_builds_only_the_selected_target() -> None:
    """Build the controller with the docker adapter, still without importing boto3."""
    on = (
        'load_settings(server={"transport": "stdio"}, deployment={"role": "api"}, '
        'gpu_worker={"on_demand": {"enabled": True, "target": "docker"}})'
    )
    loaded = _imports_after_building(on)
    assert loaded[0] == "controller"
    assert "app.adapters.compute.docker.target" in loaded
    assert not any(module.startswith(("boto3", "app.adapters.compute.aws_ecs")) for module in loaded)


@pytest.mark.parametrize("role", ["all", "worker"])
def test_enabled_requires_the_api_role(role: str) -> None:
    """Refuse on-demand start outside the api role; one all process cannot start or stop itself."""
    with pytest.raises(ValidationError, match=r"requires deployment\.role=api"):
        _settings(role, enabled=True, target="docker")


def test_enabled_requires_a_target() -> None:
    """Refuse target none when enabled and point at dry_run."""
    with pytest.raises(ValidationError, match="dry_run"):
        _settings(enabled=True, target="none")


def test_dry_run_still_needs_a_target() -> None:
    """Accept a dry run once a target is configured."""
    assert _settings(enabled=True, target="docker", dry_run=True).gpu_worker.on_demand.dry_run


def test_docker_target_requires_a_container() -> None:
    """Require docker.container for target docker."""
    with pytest.raises(ValidationError, match=r"docker\.container"):
        _settings(target="docker", docker={"container": None})


@pytest.mark.parametrize("missing", ["region", "cluster", "service"])
def test_ecs_target_requires_region_cluster_and_service(missing: str) -> None:
    """Require every ECS coordinate for target aws_ecs."""
    ecs = {"region": "eu-central-1", "cluster": "voice", "service": "gpu-worker", missing: None}
    with pytest.raises(ValidationError, match="aws_ecs"):
        _settings(target="aws_ecs", aws_ecs=ecs)


@pytest.mark.parametrize(
    ("window", "message"),
    [
        (_window(start="10:00", end="08:00"), "must be before end"),
        (_window(start="22:00", end="02:00"), "must be before end"),
        (_window(lead_minutes=-1), "greater than or equal to 0"),
        (_window(preload=["gpu:large"]), "kind:name"),
        (_window(preload=["stt"]), "kind:name"),
        (_window(days=[]), "at least 1"),
        (_window(days=["monday"]), "mon"),
    ],
    ids=["start after end", "spans midnight", "negative lead", "unknown kind", "no name", "no days", "bad day"],
)
def test_malformed_windows_fail_startup(window: dict[str, Any], message: str) -> None:
    """Reject windows that are empty, inverted, span midnight or name malformed preload entries."""
    with pytest.raises(ValidationError, match=message):
        _settings(schedule={"windows": [window]})


def test_unknown_timezone_fails_startup() -> None:
    """Require an IANA timezone name."""
    with pytest.raises(ValidationError, match="IANA"):
        _settings(schedule={"timezone": "Mars/Olympus"})


def test_windows_parse_into_domain_values() -> None:
    """Parse days, times and preload entries at load time."""
    settings = _settings(
        schedule={
            "timezone": "Europe/Zagreb",
            "windows": [_window(days=["mon", "fri"], lead_minutes=10, preload=["stt:large-v3"])],
        }
    )
    window = settings.gpu_worker.on_demand.schedule.to_domain().windows[0]
    assert (window.days, window.start, window.end) == (frozenset({0, 4}), time(8), time(10))
    assert window.preload == (ModelRef(ModelKind.STT, "large-v3"),)


def test_short_aliases_set_role_and_master_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map SERVICE_ROLE and GPU_ON_DEMAND to deployment.role and gpu_worker.on_demand.enabled."""
    monkeypatch.setenv("SERVICE_ROLE", "api")
    monkeypatch.setenv("GPU_ON_DEMAND", "true")
    monkeypatch.setenv("VOICE_MCP_GPU_WORKER__ON_DEMAND__TARGET", "docker")
    settings = load_settings(server={"transport": "stdio"})
    assert settings.deployment.role == "api"
    assert settings.gpu_worker.on_demand.enabled
