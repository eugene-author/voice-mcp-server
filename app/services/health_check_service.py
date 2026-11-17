"""Health checks for the MCP health_check tool and deeper diagnostics for the `diagnose` CLI command."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.domain.errors import CapabilityUnavailableError, VoiceMcpError
from app.domain.host import HostProfile
from app.domain.stt import SttBackendName, SttRequest, VadSettings
from app.domain.workers import CapabilitySnapshot
from app.ports.models import ModelCatalog
from app.ports.runtime import RuntimeCatalog
from app.services.model_host import ModelHost

logger = logging.getLogger(__name__)
CHECK_TIMEOUT_SECONDS = 5.0
PROBE_SILENCE_FILE = "diagnose-silence.wav"
PROBE_SAMPLE_RATE = 16000
MODEL_SIZE_ORDER = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")

type DependencyCheck = Callable[[], Awaitable[None]]
"""Raises when a dependency is unreachable."""


@dataclass(frozen=True, slots=True)
class GpuWorkerReport:
    """The gpu_worker section of health_check: role, on-demand settings, target and worker state."""

    details: dict[str, Any]
    gpu_stopped: bool = False
    """True while the compute target is stopped, with or without demand; health then reads degraded, not failed."""

    warnings: tuple[str, ...] = ()
    """Added to details.warnings: why work is waiting for the GPU worker."""


type GpuWorkerStatus = Callable[[], Awaitable[GpuWorkerReport]]


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Result of the lightweight health check."""

    status: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Result of a diagnostic run: host facts, backend selection, models, dependencies and an optional STT probe."""

    profile: str
    host: dict[str, Any]
    selection: dict[str, Any]
    data_root: str
    installed_models: list[str]
    dependencies: dict[str, str]
    stt_probe: dict[str, Any] | None
    """None when inference was not requested."""

    def to_dict(self) -> dict[str, Any]:
        """Return the report as JSON-serializable data."""
        report: dict[str, Any] = {
            "profile": self.profile,
            "host": self.host,
            "selection": self.selection,
            "data_root": self.data_root,
            "installed_models": self.installed_models,
            "dependencies": self.dependencies,
        }
        if self.stt_probe is not None:
            report["stt_probe"] = self.stt_probe
        return report


class HealthCheckService:
    """Reports health without touching models, and runs on-demand diagnostics that may load one STT model."""

    def __init__(
        self,
        *,
        host: HostProfile,
        runtime: RuntimeCatalog,
        catalog: ModelCatalog,
        checks: Mapping[str, DependencyCheck],
        worker_state: Callable[[], str],
        data_root: Path,
        workspace_root: Path,
        model_host: ModelHost | None = None,
        gpu_worker: GpuWorkerStatus | None = None,
        notes: tuple[str, ...] = (),
    ) -> None:
        """Store host facts, the runtime catalog, named dependency checks and, in-process only, the model_host."""
        self._host = host
        self._runtime = runtime
        self._catalog = catalog
        self._checks = checks
        self._worker_state = worker_state
        self._model_host = model_host
        self._gpu_worker = gpu_worker
        self._data_root = data_root
        self._workspace_root = workspace_root
        self._notes = notes

    async def check(self) -> HealthReport:
        """Run dependency checks concurrently; does not load weights or wait for the inference executor."""
        dependencies = await self._run_checks()
        healthy = all(result == "ok" for result in dependencies.values())
        warnings = list(self._notes)
        try:
            snapshot: CapabilitySnapshot | None = self._runtime.snapshot()
        except CapabilityUnavailableError as exc:
            snapshot, healthy = None, False
            warnings.append(exc.message)
        if snapshot is not None and snapshot.tts_device == "cpu":
            warnings.append("TTS runs on CPU and will be slow")
        details: dict[str, Any] = {
            **_runtime_details(snapshot),
            "worker": self._worker_state(),
            "dependencies": dependencies,
            "host": self._host_summary(),
        }
        if self._gpu_worker is not None:
            gpu = await self._gpu_worker()
            details["gpu_worker"] = gpu.details
            healthy = healthy and not gpu.gpu_stopped
            warnings.extend(gpu.warnings)
        if warnings:
            details["warnings"] = warnings
        return HealthReport(status="ok" if healthy else "degraded", details=details)

    async def run_diagnostics(self, *, include_inference: bool = True) -> DiagnosticReport:
        """Report host and runtime facts, installed models and dependencies, plus a short STT probe if requested."""
        host = self._host
        snapshot = self._runtime.snapshot()
        runtime = snapshot.stt_runtime
        return DiagnosticReport(
            profile=snapshot.profile,
            host={
                **self._host_summary(),
                "ctranslate2_cuda_devices": host.cuda_device_count,
                "ctranslate2_cuda_compute_types": sorted(host.cuda_compute_types),
                "torch_cuda": host.torch_cuda_available,
                "extras": {
                    "qwen_tts": host.qwen_tts_available,
                    "mlx_whisper": host.mlx_available,
                    "mlx_audio": host.mlx_audio_available,
                    "deepfilternet": host.denoise_runtime_available,
                },
            },
            selection={
                "stt": str(runtime.backend),
                "stt_device": runtime.device,
                "stt_compute_type": runtime.compute_type,
                "tts": str(snapshot.tts_backend) if snapshot.tts_backend else "unavailable",
                "tts_device": snapshot.tts_device,
                "denoise_device": snapshot.denoise_device,
            },
            data_root=str(self._data_root),
            installed_models=[f"{m.kind}/{m.backend}/{m.name}" for m in self._catalog.installed()],
            dependencies=await self._run_checks(),
            stt_probe=await self._stt_probe() if include_inference else None,
        )

    async def _stt_probe(self) -> dict[str, Any]:
        """Transcribe one second of silence with the smallest installed model for the active STT backend."""
        model_host = self._model_host
        if model_host is None:
            return {"status": "skipped", "reason": "inference runs in a separate GPU worker"}
        backend = model_host.stt.name
        names = sorted(model_host.installed_stt_names(), key=model_rank)
        if not names:
            return {"status": "skipped", "reason": f"no STT model installed for {backend}"}
        model = model_host.stt_model(names[0])
        silence = self._workspace_root / PROBE_SILENCE_FILE
        await asyncio.to_thread(
            sf.write, str(silence), np.zeros(PROBE_SAMPLE_RATE, dtype=np.float32), PROBE_SAMPLE_RATE
        )
        started = time.perf_counter()
        try:
            await model_host.transcribe(
                model,
                SttRequest(
                    audio_path=silence,
                    language="en",
                    beam_size=1,
                    temperature=0.0,
                    word_timestamps=False,
                    vad=VadSettings(enabled=backend is SttBackendName.CTRANSLATE2),
                ),
            )
        except VoiceMcpError as exc:
            return {"status": "failed", "model": model.name, "error": exc.message}
        finally:
            silence.unlink(missing_ok=True)
        return {"status": "ok", "model": model.name, "seconds": round(time.perf_counter() - started, 2)}

    async def _run_checks(self) -> dict[str, str]:
        """Run every dependency check concurrently and return ok or a short failure per name."""
        names = list(self._checks)
        results = await asyncio.gather(*(self._run(self._checks[name]) for name in names))
        return dict(zip(names, results, strict=True))

    def _host_summary(self) -> dict[str, Any]:
        """Return the host facts shared by both reports."""
        return {
            "system": self._host.system,
            "machine": self._host.machine,
            "cpu_count": self._host.cpu_count,
            "ram_mb": self._host.total_ram_mb,
            "gpus": [{"name": gpu.name, "memory_mb": gpu.memory_mb} for gpu in self._host.gpus],
        }

    @staticmethod
    async def _run(check: DependencyCheck) -> str:
        """Return ok or a short failure description; this is a boundary that contains any check failure."""
        try:
            await asyncio.wait_for(check(), timeout=CHECK_TIMEOUT_SECONDS)
        except TimeoutError:
            return "timeout"
        except VoiceMcpError as exc:
            return f"error: {exc.message}"
        except Exception as exc:
            logger.warning("health dependency check failed", extra={"error": type(exc).__name__})
            return f"error: {type(exc).__name__}"
        return "ok"


def _runtime_details(snapshot: CapabilitySnapshot | None) -> dict[str, Any]:
    """Return the profile, runtimes, devices and resident models of the health report."""
    if snapshot is None:
        return {"profile": None, "stt": None, "tts": None, "denoise": None, "loaded_models": {}}
    runtime = snapshot.stt_runtime
    return {
        "profile": snapshot.profile,
        "stt": {"backend": str(runtime.backend), "device": runtime.device, "compute_type": runtime.compute_type},
        "tts": (
            {"backend": str(snapshot.tts_backend), "device": snapshot.tts_device} if snapshot.tts_backend else None
        ),
        "denoise": (
            {"engine": "deepfilternet", "device": snapshot.denoise_device} if snapshot.denoise_device else None
        ),
        "loaded_models": dict(snapshot.loaded),
    }


def model_rank(name: str) -> int:
    """Order Whisper model names from smallest to largest; unknown names sort last."""
    return MODEL_SIZE_ORDER.index(name) if name in MODEL_SIZE_ORDER else len(MODEL_SIZE_ORDER)
