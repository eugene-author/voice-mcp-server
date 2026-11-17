"""HealthCheckService: the lightweight MCP health check and the diagnose workflow, including resource cleanup."""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from app import main
from app.core.settings import DatabaseSettings, Settings
from app.domain.errors import InferenceError
from app.domain.models import ModelKind
from app.main import InferenceComponents, diagnostics_session
from app.services.health_check_service import GpuWorkerReport, HealthCheckService, model_rank
from tests.harness import Harness, build_harness, host_profile
from tests.mocks.stt import MockSttBackend


def _service(harness: Harness, **checks: object) -> HealthCheckService:
    """Return a health service over the harness components."""
    return HealthCheckService(
        host=host_profile(),
        runtime=harness.runtime,
        catalog=harness.model_host._catalog,
        checks=checks or {"storage": harness.store.verify},  # type: ignore[arg-type]
        worker_state=lambda: "active",
        data_root=harness.settings.data_root,
        workspace_root=harness.settings.workspace_root,
        model_host=harness.model_host,
    )


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Return a harness with its workspace root created."""
    harness = build_harness(tmp_path)
    harness.settings.workspace_root.mkdir(parents=True, exist_ok=True)
    yield harness
    harness.executor.shutdown()


async def test_check_never_loads_models(harness: Harness) -> None:
    """Report health without loading weights or using the inference thread."""
    report = await _service(harness).check()
    assert report.status == "ok"
    assert report.details["dependencies"] == {"storage": "ok"}
    assert harness.stt.load_count == 0


async def test_diagnostics_without_inference_load_nothing(harness: Harness) -> None:
    """Map --skip-inference to include_inference=False: facts and dependencies only, no probe and no weights."""
    report = await _service(harness).run_diagnostics(include_inference=False)
    data = report.to_dict()
    assert "stt_probe" not in data
    assert data["selection"]["stt"] == "ctranslate2"
    assert "stt/ctranslate2/small" in data["installed_models"]
    assert data["dependencies"] == {"storage": "ok"}
    assert harness.stt.load_count == 0


async def test_diagnostics_probe_uses_smallest_installed_model(tmp_path: Path) -> None:
    """Transcribe a second of silence with the smallest installed model and clean up the probe file."""
    harness = build_harness(
        tmp_path, installed={(ModelKind.STT, "ctranslate2", "medium"), (ModelKind.STT, "ctranslate2", "tiny")}
    )
    harness.settings.workspace_root.mkdir(parents=True, exist_ok=True)
    report = await _service(harness).run_diagnostics()
    assert report.stt_probe is not None
    assert report.stt_probe["status"] == "ok"
    assert report.stt_probe["model"] == "tiny"
    assert harness.stt.load_count == 1
    assert list(harness.settings.workspace_root.glob("diagnose-*")) == []
    harness.executor.shutdown()


async def test_diagnostics_without_stt_model_reports_skip(tmp_path: Path) -> None:
    """Explain that the probe was skipped when no model is installed for the active backend."""
    harness = build_harness(tmp_path, installed={(ModelKind.STT, "mlx", "small")})
    report = await _service(harness).run_diagnostics()
    assert report.stt_probe == {"status": "skipped", "reason": "no STT model installed for ctranslate2"}
    harness.executor.shutdown()


async def test_diagnostics_reports_probe_failure(tmp_path: Path) -> None:
    """Report a failing probe as a result rather than crashing the command."""
    harness = build_harness(tmp_path, stt=MockSttBackend(error=InferenceError("cuda exploded")))
    harness.settings.workspace_root.mkdir(parents=True, exist_ok=True)
    report = await _service(harness).run_diagnostics()
    assert report.stt_probe == {"status": "failed", "model": "small", "error": "cuda exploded"}
    harness.executor.shutdown()


async def test_failed_dependency_is_reported_with_its_message(harness: Harness) -> None:
    """Name the failing dependency and the reason instead of raising."""

    async def missing() -> None:
        """Fail like an unset DATABASE_URL."""
        await main._database_not_configured()

    report = await _service(harness, database=missing).run_diagnostics(include_inference=False)
    assert report.dependencies == {"database": "error: DATABASE_URL is not set"}


def _components(harness: Harness) -> InferenceComponents:
    """Return inference components built from the harness."""
    return InferenceComponents(
        host=host_profile(),
        stt=harness.stt,
        tts=harness.tts,
        denoiser=None,
        model_host=harness.model_host,
        runtime=harness.local_runtime,
        executor=harness.executor,
        denoise_device=None,
        profile="linux-cpu",
    )


@pytest.mark.parametrize("probe_result", ["success", "failure", "cancelled"])
async def test_diagnostics_session_releases_resources(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, probe_result: str
) -> None:
    """Unload models and stop the inference thread after a successful, failed or cancelled diagnose run."""
    monkeypatch.setattr(main, "build_inference", lambda _settings: _components(harness))
    monkeypatch.setattr(main, "enforce_offline_runtime", lambda: None)
    settings: Settings = harness.settings.model_copy(update={"database": DatabaseSettings()})

    async def run() -> None:
        """Run diagnostics with a probe, failing afterwards when requested."""
        async with diagnostics_session(settings) as health:
            report = await health.run_diagnostics()
            assert report.stt_probe is not None
            assert report.dependencies == {"database": "error: DATABASE_URL is not set"}
            if probe_result == "failure":
                raise RuntimeError("diagnostics interrupted")
            if probe_result == "cancelled":
                raise asyncio.CancelledError

    if probe_result == "failure":
        with pytest.raises(RuntimeError, match="interrupted"):
            await run()
    elif probe_result == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await run()
    else:
        await run()
    assert harness.stt.loaded is None
    with pytest.raises(RuntimeError):
        await harness.executor.run(lambda: None)


def test_model_rank_orders_by_size() -> None:
    """Sort Whisper names from smallest to largest with unknown names last."""
    assert sorted(["large-v3", "custom", "tiny", "small"], key=model_rank) == ["tiny", "small", "large-v3", "custom"]


async def test_gpu_worker_section_is_additive_and_degrades_while_the_gpu_is_stopped(harness: Harness) -> None:
    """Add the gpu_worker section and its warnings, and read degraded (not failed) while the GPU is stopped."""
    stopped = {"role": "api", "on_demand": True, "target": "docker", "dry_run": False, "target_state": "stopped"}

    async def gpu_worker() -> GpuWorkerReport:
        """Report a GPU kept stopped by the wake controller."""
        return GpuWorkerReport(stopped, gpu_stopped=True, warnings=("GPU worker is stopped; 2 jobs waiting",))

    service = HealthCheckService(
        host=host_profile(),
        runtime=harness.runtime,
        catalog=harness.model_host._catalog,
        checks={"storage": harness.store.verify},
        worker_state=lambda: "remote",
        data_root=harness.settings.data_root,
        workspace_root=harness.settings.workspace_root,
        gpu_worker=gpu_worker,
    )
    report = await service.check()
    assert report.status == "degraded"
    assert report.details["gpu_worker"] == stopped
    assert report.details["dependencies"] == {"storage": "ok"}
    assert "GPU worker is stopped; 2 jobs waiting" in report.details["warnings"]
