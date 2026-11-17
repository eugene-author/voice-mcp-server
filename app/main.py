"""Composition root: builds the object graph once from Settings and runs the selected transport."""

import asyncio
import functools
import logging
import os
import signal
import socket
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import httpx
from mcp.server.auth.provider import TokenVerifier
from mcp.server.mcpserver import MCPServer
from sqlalchemy.ext.asyncio import AsyncEngine

from app.adapters.denoise.factory import build_denoiser
from app.adapters.host.probe import SystemHostProbe, os_user
from app.adapters.identity.oidc.factory import build_token_verifier, key_endpoint
from app.adapters.media.ffmpeg.tool import FfmpegMediaTool
from app.adapters.models.factory import build_model_catalog
from app.adapters.queue.asyncio.queue import AsyncioJobQueue, AsyncioJobWaiters
from app.adapters.queue.postgres_notify.queue import (
    Connect,
    PostgresAnnouncingJobQueue,
    PostgresListeningJobQueue,
    connector,
)
from app.adapters.storage.factory import build_asset_store
from app.adapters.stt.factory import build_stt_backend
from app.adapters.tts.factory import build_tts_engine
from app.adapters.vad.factory import build_vad
from app.core.clock import SystemClock, SystemMonotonicClock
from app.core.executor import InferenceRunner
from app.core.logging import configure_logging
from app.core.settings import Settings, load_presets
from app.db.engine import create_engine, create_session_factory, plain_database_url
from app.db.locks import CONTROLLER_LOCK_KEY, PostgresWorkerLock
from app.db.unit_of_work import SqlUnitOfWorkFactory
from app.domain.assets import StorageBackend
from app.domain.errors import ConfigurationError, VoiceMcpError
from app.domain.host import HostProfile
from app.domain.jobs import JobType
from app.domain.stt import SttBackendName, SttRuntime
from app.domain.workers import ModelRef, WakePolicy
from app.mcp_server.auth import CallerResolver
from app.mcp_server.dependencies import ToolServices
from app.mcp_server.server import build_mcp_server, transport_security
from app.ports.denoise import Denoiser
from app.ports.job_queue import JobQueue
from app.ports.runtime import RuntimeCatalog
from app.ports.storage import AssetStore
from app.ports.stt import SttBackend
from app.ports.tts import TtsEngine
from app.ports.unit_of_work import UnitOfWorkFactory
from app.services.batch_service import BatchService
from app.services.health_check_service import DependencyCheck, GpuWorkerReport, GpuWorkerStatus, HealthCheckService
from app.services.inference_progress import (
    STT_SECONDS_PER_AUDIO_SECOND,
    TTS_SECONDS_PER_CHARACTER,
    InferenceProgress,
)
from app.services.input_policy import InputPathPolicy
from app.services.input_transfer import InputUploader
from app.services.job_planning import JobPlanner
from app.services.job_query_service import JobQueryService
from app.services.job_service import JobService
from app.services.job_workspace import JobWorkspace
from app.services.model_host import ModelHost
from app.services.model_service import ModelService
from app.services.publishing import AssetPublisher
from app.services.runtime_catalog import LocalRuntimeCatalog, RegistryRuntimeCatalog
from app.services.runtime_selection import (
    exclusive_residency,
    platform_profile,
    select_ct2_runtime,
    select_stt_backend,
    select_torch_device,
    select_tts_backend,
)
from app.services.staged_job_assets import StagedJobAssets
from app.services.synthesis_service import SynthesisService
from app.services.transcription_options_resolver import TranscriptionOptionsResolver
from app.services.transcription_service import TranscriptionService
from app.services.voice_registration_service import VoiceRegistrationService
from app.services.voice_service import VoiceService
from app.workers.gpu_operator import GpuOperator
from app.workers.gpu_worker import GpuWorker
from app.workers.heartbeat import Heartbeat
from app.workers.idle_unload import IdleUnloader
from app.workers.in_process_worker import InProcessWorker
from app.workers.job_transitions import JobTransitions
from app.workers.job_worker import JobHandler, JobWorker
from app.workers.reconciliation import AssetReconciler
from app.workers.recovery import JobRecovery
from app.workers.wake_controller import WakeController

logger = logging.getLogger(__name__)
RECONCILE_INTERVAL_SECONDS = 3600
DEFAULT_REGISTRY_REFRESH_SECONDS = 15.0
OFFLINE_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
try:
    APP_VERSION = version("voice-mcp-server")
except PackageNotFoundError:
    APP_VERSION = "unknown"

type Loop = Callable[[], Coroutine[None, None, None]]


@dataclass(frozen=True, slots=True)
class InferenceComponents:
    """Inference components that do not need the database."""

    host: HostProfile
    stt: SttBackend
    tts: TtsEngine | None
    denoiser: Denoiser | None
    model_host: ModelHost
    runtime: LocalRuntimeCatalog
    executor: InferenceRunner
    denoise_device: str | None
    profile: str
    """Platform profile derived from the devices actually selected."""


@dataclass(frozen=True, slots=True)
class Container:
    """Application object graph, built once at startup."""

    settings: Settings
    inference: InferenceComponents | None
    """None in the api role, which loads no models."""

    engine: AsyncEngine
    uow: UnitOfWorkFactory
    store: AssetStore
    workspace: JobWorkspace
    token_verifier: TokenVerifier | None
    """SDK TokenVerifier implemented by JwtTokenVerifier; None in stdio mode."""

    tools: ToolServices
    voices: VoiceService
    background: Loop
    """The in-process worker (all role), or recovery, reconciliation, catalog refresh and announcing (api role)."""


def enforce_offline_runtime() -> None:
    """Forbid implicit model downloads by Hugging Face libraries for the rest of the process."""
    os.environ.update(OFFLINE_ENVIRONMENT)


def prepare_data_root(settings: Settings) -> None:
    """Create the runtime data directories with restrictive permissions; the api role gets no models directory."""
    directories = [settings.data_root, settings.workspace_root, settings.data_root / "logs"]
    if settings.deployment.role != "api":
        directories.append(settings.models_root)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True, mode=0o750)
    if settings.storage.backend == "local":
        settings.assets_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    if not settings.server.allowed_input_directories:
        (settings.data_root / "inputs").mkdir(parents=True, exist_ok=True, mode=0o750)


def stt_runtime_for(name: SttBackendName, settings: Settings, host: HostProfile) -> SttRuntime:
    """Return the runtime of the selected STT backend."""
    if name is SttBackendName.MLX:
        return SttRuntime(backend=SttBackendName.MLX, device="metal", compute_type="float16")
    return select_ct2_runtime(
        host,
        device=settings.stt.device,
        compute_type=settings.stt.compute_type,
        low_vram_threshold_mb=settings.stt.low_vram_threshold_mb,
    )


def build_inference(settings: Settings) -> InferenceComponents:
    """Probe the host and construct the inference components; missing extras leave components unset."""
    host = SystemHostProbe().probe()
    stt_name = select_stt_backend(settings.stt.backend, host)
    stt = build_stt_backend(stt_name, stt_runtime_for(stt_name, settings, host), settings, host)
    tts_name = select_tts_backend(settings.tts.backend, host)
    tts = None
    if tts_name is not None:
        tts_device = "cpu" if host.is_apple_silicon else select_torch_device(host, allow_cpu=settings.tts.allow_cpu)
        tts = build_tts_engine(
            tts_name, device=tts_device, conditioning_cache_size=settings.tts.conditioning_cache_size
        )
    denoise_device: str | None = None
    denoiser = None
    if host.denoise_runtime_available:
        requested: Literal["cuda", "cpu"] = "cuda" if host.torch_cuda_available and not host.is_apple_silicon else "cpu"
        denoiser = build_denoiser(device=requested, attenuation_limit_db=settings.audio.denoising.attenuation_limit_db)
        denoise_device = denoiser.device
    executor = InferenceRunner()
    catalog = build_model_catalog(settings)
    model_host = ModelHost(
        stt=stt,
        tts=tts,
        denoiser=denoiser,
        vad=build_vad() if stt_name is SttBackendName.MLX else None,
        catalog=catalog,
        executor=executor,
        exclusive=exclusive_residency(host, ram_threshold_mb=settings.hardware.exclusive_residency_ram_mb),
        denoise_model=settings.audio.denoising.model,
    )
    profile = platform_profile(host, devices=(stt.runtime.device, tts.device if tts else None, denoise_device))
    return InferenceComponents(
        host=host,
        stt=stt,
        tts=tts,
        denoiser=denoiser,
        model_host=model_host,
        runtime=LocalRuntimeCatalog(
            model_host=model_host, catalog=catalog, profile=profile, denoise_device=denoise_device
        ),
        executor=executor,
        denoise_device=denoise_device,
        profile=profile,
    )


def _database_check(uow: UnitOfWorkFactory) -> DependencyCheck:
    """Return a check that runs one trivial query."""

    async def check() -> None:
        """Query the jobs table."""
        async with uow() as work:
            await work.jobs.queued_ids(limit=1)

    return check


async def _database_not_configured() -> None:
    """Fail the database check when DATABASE_URL is unset."""
    raise ConfigurationError("DATABASE_URL is not set")


def _idp_check(url: str, timeout_seconds: float) -> DependencyCheck:
    """Return a check that the identity-provider endpoint the verifier uses is reachable."""

    async def check() -> None:
        """Fetch the JWKS or discovery document."""
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()

    return check


def _health_service(
    settings: Settings,
    *,
    host: HostProfile,
    runtime: RuntimeCatalog,
    model_host: ModelHost | None,
    checks: dict[str, DependencyCheck],
    worker_state: Callable[[], str],
    gpu_worker: GpuWorkerStatus | None = None,
) -> HealthCheckService:
    """Build the health service shared by the health_check tool and the diagnose command."""
    return HealthCheckService(
        gpu_worker=gpu_worker,
        host=host,
        runtime=runtime,
        catalog=build_model_catalog(settings),
        checks=checks,
        worker_state=worker_state,
        data_root=settings.data_root,
        workspace_root=settings.workspace_root,
        model_host=model_host,
    )


async def _release_inference(inference: InferenceComponents, *, abandoned: bool = False) -> None:
    """Unload models and stop the inference thread even if unloading fails; do not wait for an abandoned native call."""
    if abandoned:
        inference.executor.shutdown(wait=False)
        return
    try:
        await inference.model_host.unload_all()
    finally:
        inference.executor.shutdown()


@asynccontextmanager
async def diagnostics_session(settings: Settings) -> AsyncIterator[HealthCheckService]:
    """Yield a health service for the diagnose command; models, the executor and the engine are always released."""
    enforce_offline_runtime()
    prepare_data_root(settings)
    async with AsyncExitStack() as stack:
        inference = build_inference(settings)
        stack.push_async_callback(_release_inference, inference)
        database_check: DependencyCheck = _database_not_configured
        if settings.database.url is not None:
            engine = create_engine(settings.database)
            stack.push_async_callback(engine.dispose)
            database_check = _database_check(SqlUnitOfWorkFactory(create_session_factory(engine)))
        yield _health_service(
            settings,
            host=inference.host,
            runtime=inference.runtime,
            model_host=inference.model_host,
            checks={"database": database_check},
            worker_state=lambda: "none",
        )


def _job_handlers(
    settings: Settings,
    *,
    model_host: ModelHost,
    uow: UnitOfWorkFactory,
    store: AssetStore,
    media: FfmpegMediaTool,
    publisher: AssetPublisher,
    workspace: JobWorkspace,
) -> dict[JobType, JobHandler]:
    """Map each executable job type to the service that runs it."""
    monotonic = SystemMonotonicClock()
    tts_progress = InferenceProgress(monotonic=monotonic, seconds_per_unit=TTS_SECONDS_PER_CHARACTER)
    return {
        JobType.SPEECH_TO_TEXT: TranscriptionService(
            probe=media,
            converter=media,
            model_host=model_host,
            publisher=publisher,
            staged=StagedJobAssets(workspace),
            workspace=workspace,
            fillers=settings.audio.fillers,
            max_duration_seconds=settings.audio.max_duration_seconds,
            text_preview_chars=settings.jobs.text_preview_chars,
            inference_progress=InferenceProgress(monotonic=monotonic, seconds_per_unit=STT_SECONDS_PER_AUDIO_SECOND),
        ),
        JobType.TEXT_TO_SPEECH: SynthesisService(
            uow=uow,
            store=store,
            converter=media,
            model_host=model_host,
            publisher=publisher,
            staged=StagedJobAssets(workspace),
            workspace=workspace,
            cloning_checkpoint=settings.tts.cloning_checkpoint,
            preset_checkpoint=settings.tts.preset_checkpoint,
            inference_progress=tts_progress,
        ),
        JobType.REGISTER_VOICE: VoiceRegistrationService(
            probe=media,
            converter=media,
            model_host=model_host,
            publisher=publisher,
            workspace=workspace,
            cloning_checkpoint=settings.tts.cloning_checkpoint,
            verification_stt_model=settings.stt.model,
            inference_progress=tts_progress,
        ),
    }


def _transitions(settings: Settings, *, uow: UnitOfWorkFactory, clock: SystemClock) -> JobTransitions:
    """Return the job transitions with the configured retry policy."""
    return JobTransitions(
        uow=uow,
        clock=clock,
        retry_backoff_seconds=settings.jobs.retry_backoff_seconds,
        save_events=settings.observability.save_job_events,
    )


def _job_worker(
    settings: Settings,
    *,
    uow: UnitOfWorkFactory,
    queue: JobQueue,
    waiters: AsyncioJobWaiters,
    handlers: dict[JobType, JobHandler],
    batches: BatchService,
    workspace: JobWorkspace,
    clock: SystemClock,
    lease_owner: str,
) -> JobWorker:
    """Return the sequential job worker with its lease heartbeat."""
    return JobWorker(
        transitions=_transitions(settings, uow=uow, clock=clock),
        handlers=handlers,
        queue=queue,
        waiters=waiters,
        batches=batches,
        workspace=workspace,
        heartbeat=Heartbeat(uow=uow, clock=clock, interval_seconds=settings.jobs.heartbeat_seconds),
        lease_owner=lease_owner,
    )


def _recovery(
    settings: Settings, *, uow: UnitOfWorkFactory, queue: JobQueue, batches: BatchService, clock: SystemClock
) -> JobRecovery:
    """Return recovery for stale leases, due retries, interrupted jobs and batch parents."""
    return JobRecovery(
        uow=uow,
        queue=queue,
        batches=batches,
        clock=clock,
        lease_timeout_seconds=settings.jobs.lease_timeout_seconds,
        poll_interval_seconds=settings.jobs.poll_interval_seconds,
    )


def _reconciler(settings: Settings, *, uow: UnitOfWorkFactory, store: AssetStore, clock: SystemClock) -> Loop:
    """Return the orphan-asset and stopped-worker reconciliation loop."""
    return AssetReconciler(
        uow=uow,
        store=store,
        clock=clock,
        retention_hours=settings.storage.orphan_retention_hours,
        interval_seconds=RECONCILE_INTERVAL_SECONDS,
    ).run


def _idle_loops(settings: Settings, model_host: ModelHost) -> tuple[Loop, ...]:
    """Return the idle-unload loop when gpu_worker.idle_unload_seconds is set."""
    idle_seconds = settings.gpu_worker.idle_unload_seconds
    return (IdleUnloader(model_host=model_host, idle_seconds=idle_seconds).run,) if idle_seconds > 0 else ()


def _in_process_worker(
    settings: Settings,
    *,
    engine: AsyncEngine,
    uow: UnitOfWorkFactory,
    store: AssetStore,
    queue: AsyncioJobQueue,
    waiters: AsyncioJobWaiters,
    handlers: dict[JobType, JobHandler],
    batches: BatchService,
    workspace: JobWorkspace,
    clock: SystemClock,
    model_host: ModelHost,
) -> InProcessWorker:
    """Assemble the in-process worker, recovery polling, reconciliation and idle unload of the all role."""
    return InProcessWorker(
        lock=PostgresWorkerLock(engine),
        worker=_job_worker(
            settings,
            uow=uow,
            queue=queue,
            waiters=waiters,
            handlers=handlers,
            batches=batches,
            workspace=workspace,
            clock=clock,
            lease_owner=lease_owner(),
        ),
        recovery=_recovery(settings, uow=uow, queue=queue, batches=batches, clock=clock),
        startup_recovery=settings.jobs.startup_recovery,
        extra_loops=(_reconciler(settings, uow=uow, store=store, clock=clock), *_idle_loops(settings, model_host)),
    )


def lease_owner() -> str:
    """Return this process's lease owner and worker ID, host:pid."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _notify_connect(settings: Settings) -> Connect:
    """Return the factory for the dedicated LISTEN/NOTIFY connections."""
    if settings.database.url is None:
        raise ConfigurationError("DATABASE_URL is not set")
    return connector(
        plain_database_url(settings.database.url.get_secret_value()),
        timeout_seconds=settings.database.connect_timeout_seconds,
    )


def _registry_refresh_seconds(settings: Settings) -> float:
    """Return how often the api role re-reads the worker's snapshot."""
    on_demand = settings.gpu_worker.on_demand
    return on_demand.tick_seconds if on_demand.enabled else DEFAULT_REGISTRY_REFRESH_SECONDS


def _remote_worker_state() -> str:
    """Return the worker state the api role reports: inference runs in a separate process."""
    return "remote"


async def _run_loops(loops: tuple[Loop, ...]) -> None:
    """Run background loops together until cancelled."""
    async with asyncio.TaskGroup() as group:
        for index, loop in enumerate(loops):
            group.create_task(loop(), name=f"background-{index}")


def _recovery_loop(recovery: JobRecovery, *, startup: bool) -> Loop:
    """Return a loop that recovers once at startup (if enabled) and then polls."""

    async def run() -> None:
        """Recover, then poll until cancelled."""
        if startup:
            try:
                await recovery.recover()
            except Exception:
                logger.exception("startup recovery failed; polling will retry")
        await recovery.poll_forever()

    return run


def wake_policy(settings: Settings) -> WakePolicy:
    """Return the wake controller's decision timings from settings."""
    on_demand = settings.gpu_worker.on_demand
    return WakePolicy(
        idle_timeout=timedelta(seconds=on_demand.idle_timeout_seconds),
        min_uptime=timedelta(seconds=on_demand.min_uptime_seconds),
        start_timeout=timedelta(seconds=on_demand.start_timeout_seconds),
        drain_timeout=timedelta(seconds=settings.gpu_worker.drain_timeout_seconds),
        drain_confirm_margin=timedelta(seconds=on_demand.drain_confirm_margin_seconds),
        lease_timeout=timedelta(seconds=settings.jobs.lease_timeout_seconds),
    )


def build_wake_controller(
    settings: Settings,
    *,
    engine: AsyncEngine,
    uow: UnitOfWorkFactory,
    clock: SystemClock,
    submitted: asyncio.Event,
    stack: AsyncExitStack,
) -> WakeController | None:
    """Return the wake controller, or None with the master switch off: then no compute adapter is even imported."""
    on_demand = settings.gpu_worker.on_demand
    if not on_demand.enabled:
        return None
    from app.adapters.compute.factory import build_compute_target

    target = build_compute_target(settings)
    stack.push_async_callback(target.aclose)
    return WakeController(
        uow=uow,
        target=target,
        lock=PostgresWorkerLock(engine, key=CONTROLLER_LOCK_KEY),
        clock=clock,
        schedule=on_demand.schedule.to_domain(),
        policy=wake_policy(settings),
        backoff_steps=on_demand.start_backoff_seconds,
        tick_seconds=on_demand.tick_seconds,
        dry_run=on_demand.dry_run,
        submitted=submitted,
    )


def _gpu_worker_status(
    settings: Settings, *, uow: UnitOfWorkFactory, controller: WakeController | None
) -> GpuWorkerStatus:
    """Return the provider of health_check's gpu_worker section."""
    on_demand = settings.gpu_worker.on_demand
    role = settings.deployment.role
    base = {"role": role, "on_demand": on_demand.enabled, "target": on_demand.target, "dry_run": on_demand.dry_run}

    async def status() -> GpuWorkerReport:
        """Read the newest worker row (api role) and the controller's latest decision."""
        details: dict[str, object] = dict(base)
        if role == "api":
            try:
                async with uow() as work:
                    worker = await work.workers.latest()
                details["worker_status"] = str(worker.status) if worker else None
            except Exception as exc:
                logger.warning("worker registry unreadable for health", extra={"error": type(exc).__name__})
                details["worker_status"] = "unknown"
        if controller is not None:
            details.update(controller.report())
        warning = controller.demand_warning if controller else None
        return GpuWorkerReport(
            dict(details),
            gpu_stopped=bool(controller and controller.gpu_stopped),
            warnings=(warning,) if warning else (),
        )

    return status


@dataclass(frozen=True, slots=True)
class _Submissions:
    """The services the MCP tools use to submit, read and manage jobs and voices."""

    jobs: JobService
    queries: JobQueryService
    voices: VoiceService
    batches: BatchService


def _submissions(
    settings: Settings,
    *,
    runtime: RuntimeCatalog,
    uow: UnitOfWorkFactory,
    store: AssetStore,
    queue: JobQueue,
    waiters: AsyncioJobWaiters,
    workspace: JobWorkspace,
    publisher: AssetPublisher,
    clock: SystemClock,
    monotonic: SystemMonotonicClock,
    submitted: asyncio.Event | None = None,
) -> _Submissions:
    """Build the planner, job, query, voice and batch services over a runtime catalog."""
    uploader = None
    if settings.storage.input_transfer == "ingest":
        uploader = InputUploader(store=store, max_batch_bytes=settings.audio.max_batch_upload_bytes)
    presets = load_presets()
    planner = JobPlanner(
        settings=settings,
        presets=presets,
        stt_options=TranscriptionOptionsResolver(settings=settings, presets=presets, runtime=runtime),
        runtime=runtime,
        input_policy=InputPathPolicy(
            allowed_directories=settings.input_directories,
            accepted_formats=settings.audio.accepted_formats,
            max_input_bytes=settings.audio.max_input_bytes,
        ),
        uow=uow,
        clock=clock,
    )
    queries = JobQueryService(
        uow=uow,
        store=store,
        default_limit=settings.jobs.list_jobs_default_limit,
        max_limit=settings.jobs.list_jobs_max_limit,
        presign_enabled=settings.storage.s3.presign_downloads,
        presign_ttl_seconds=settings.storage.s3.presign_ttl_seconds,
    )
    return _Submissions(
        jobs=JobService(
            uow=uow,
            planner=planner,
            queue=queue,
            waiters=waiters,
            queries=queries,
            runtime=runtime,
            clock=clock,
            monotonic=monotonic,
            submitted=submitted,
            uploader=uploader,
        ),
        queries=queries,
        voices=VoiceService(
            uow=uow, store=store, runtime=runtime, preset_checkpoint=settings.tts.preset_checkpoint, clock=clock
        ),
        batches=BatchService(uow=uow, publisher=publisher, workspace=workspace, waiters=waiters, clock=clock),
    )


async def _storage(settings: Settings, stack: AsyncExitStack) -> tuple[AsyncEngine, UnitOfWorkFactory, AssetStore]:
    """Create the database engine (disposed by stack) and the asset store, verifying S3 access."""
    engine = create_engine(settings.database)
    stack.push_async_callback(engine.dispose)
    store = build_asset_store(settings)
    if store.backend is StorageBackend.S3:
        await store.verify()
    return engine, SqlUnitOfWorkFactory(create_session_factory(engine)), store


async def build_container(
    settings: Settings, *, transport: Literal["streamable_http", "stdio"], stack: AsyncExitStack
) -> Container:
    """Build the all or api role for serving; stack releases what was built, also when building fails."""
    enforce_offline_runtime()
    prepare_data_root(settings)
    api_role = settings.deployment.role == "api"
    inference: InferenceComponents | None = None
    if not api_role:
        inference = build_inference(settings)
        stack.push_async_callback(_release_inference, inference)
    clock, monotonic = SystemClock(), SystemMonotonicClock()
    engine, uow, store = await _storage(settings, stack)
    workspace = JobWorkspace(settings.workspace_root)
    publisher = AssetPublisher(store)
    waiters = AsyncioJobWaiters()
    runtime: RuntimeCatalog
    queue: JobQueue
    submitted = asyncio.Event()
    controller = None
    if inference is None:
        controller = build_wake_controller(
            settings, engine=engine, uow=uow, clock=clock, submitted=submitted, stack=stack
        )
        registry = RegistryRuntimeCatalog(uow=uow, refresh_seconds=_registry_refresh_seconds(settings))
        try:
            await registry.refresh()
        except VoiceMcpError as exc:
            logger.warning("could not read the worker registry yet", extra={"error": exc.message})
        announcer = PostgresAnnouncingJobQueue(connect=_notify_connect(settings), maxsize=settings.jobs.queue_size)
        runtime, queue, host = registry, announcer, SystemHostProbe().probe_basic()
    else:
        local_queue = AsyncioJobQueue(maxsize=settings.jobs.queue_size)
        runtime, queue, host = inference.runtime, local_queue, inference.host
    services = _submissions(
        settings,
        runtime=runtime,
        uow=uow,
        store=store,
        queue=queue,
        waiters=waiters,
        workspace=workspace,
        publisher=publisher,
        clock=clock,
        monotonic=monotonic,
        submitted=submitted if controller else None,
    )
    if inference is None:
        recovery = _recovery(settings, uow=uow, queue=queue, batches=services.batches, clock=clock)
        loops = (
            announcer.run,
            registry.run,
            _recovery_loop(recovery, startup=settings.jobs.startup_recovery),
            _reconciler(settings, uow=uow, store=store, clock=clock),
            *((controller.run,) if controller else ()),
        )
        background: Loop = functools.partial(_run_loops, loops)
        worker_state: Callable[[], str] = _remote_worker_state
    else:
        in_process_worker = _in_process_worker(
            settings,
            engine=engine,
            uow=uow,
            store=store,
            queue=local_queue,
            waiters=waiters,
            handlers=_job_handlers(
                settings,
                model_host=inference.model_host,
                uow=uow,
                store=store,
                media=FfmpegMediaTool.from_path(),
                publisher=publisher,
                workspace=workspace,
            ),
            batches=services.batches,
            workspace=workspace,
            clock=clock,
            model_host=inference.model_host,
        )
        background, worker_state = in_process_worker.run, lambda: str(in_process_worker.state)
    token_verifier: TokenVerifier | None = None
    checks: dict[str, DependencyCheck] = {"database": _database_check(uow), "storage": store.verify}
    if transport == "streamable_http":
        token_verifier = await build_token_verifier(settings.auth)
        checks["identity_provider"] = _idp_check(key_endpoint(settings.auth), settings.auth.jwks_fetch_timeout_seconds)
    tools = ToolServices(
        caller_resolver=(
            CallerResolver.for_stdio(os_user()) if transport == "stdio" else CallerResolver(trusted_local=None)
        ),
        jobs=services.jobs,
        queries=services.queries,
        models=ModelService(catalog=build_model_catalog(settings), runtime=runtime),
        voices=services.voices,
        health=_health_service(
            settings,
            host=host,
            runtime=runtime,
            model_host=inference.model_host if inference else None,
            checks=checks,
            worker_state=worker_state,
            gpu_worker=_gpu_worker_status(settings, uow=uow, controller=controller),
        ),
        inline_wait_seconds=settings.jobs.inline_wait_seconds,
    )
    return Container(
        settings=settings,
        inference=inference,
        engine=engine,
        uow=uow,
        store=store,
        workspace=workspace,
        token_verifier=token_verifier,
        tools=tools,
        voices=services.voices,
        background=background,
    )


@asynccontextmanager
async def running_container(
    settings: Settings, *, transport: Literal["streamable_http", "stdio"]
) -> AsyncIterator[Container]:
    """Yield a built container; models, the executor and the engine are released afterwards or on a failed build."""
    async with AsyncExitStack() as stack:
        yield await build_container(settings, transport=transport, stack=stack)


def create_mcp_server(container: Container) -> MCPServer:
    """Create the MCP server with tools and SDK-managed OAuth settings."""
    return build_mcp_server(container.settings, container.tools, container.token_verifier)


async def serve(settings: Settings) -> None:
    """Run the MCP transport and the role's background loops until the transport stops."""
    transport = settings.server.transport
    async with running_container(settings, transport=transport) as container:
        server = create_mcp_server(container)
        inference = container.inference
        logger.info(
            "starting server",
            extra={
                "transport": transport,
                "role": settings.deployment.role,
                "profile": inference.profile if inference else None,
                "stt": str(inference.stt.name) if inference else None,
                "tts": str(inference.tts.name) if inference and inference.tts else None,
            },
        )
        async with asyncio.TaskGroup() as group:
            background = group.create_task(container.background(), name="background")
            try:
                if transport == "stdio":
                    await server.run_stdio_async()
                else:
                    await server.run_streamable_http_async(
                        host=settings.server.host,
                        port=settings.server.port,
                        transport_security=transport_security(settings.server),
                    )
            finally:
                background.cancel()


def run(settings: Settings) -> None:
    """Configure logging and run the server."""
    configure_logging(level=settings.observability.log_level, log_format=settings.observability.log_format)
    asyncio.run(serve(settings))


@asynccontextmanager
async def gpu_operator_session(settings: Settings) -> AsyncIterator[GpuOperator]:
    """Yield the operator for the gpu CLI; the engine and the compute target's client are released afterwards."""
    async with AsyncExitStack() as stack:
        engine = create_engine(settings.database)
        stack.push_async_callback(engine.dispose)
        target = None
        if settings.gpu_worker.on_demand.target != "none":
            from app.adapters.compute.factory import build_compute_target

            target = build_compute_target(settings)
            stack.push_async_callback(target.aclose)
        yield GpuOperator(
            uow=SqlUnitOfWorkFactory(create_session_factory(engine)),
            target=target,
            clock=SystemClock(),
            schedule=settings.gpu_worker.on_demand.schedule.to_domain(),
            policy=wake_policy(settings),
        )


def _window_preload(settings: Settings, now: datetime) -> tuple[ModelRef, ...]:
    """Return the preload list of a schedule window that is active or within its lead time."""
    window = settings.gpu_worker.on_demand.schedule.to_domain().active_window(now)
    return window.preload if window else ()


@asynccontextmanager
async def gpu_worker_session(settings: Settings, *, preload: tuple[ModelRef, ...] = ()) -> AsyncIterator[GpuWorker]:
    """Yield the worker role's process; models, the executor and the engine are released afterwards."""
    enforce_offline_runtime()
    prepare_data_root(settings)
    async with AsyncExitStack() as stack:
        inference = build_inference(settings)
        gpu: GpuWorker | None = None

        async def release() -> None:
            """Release inference, which leaves a native call the worker abandoned to end with the process."""
            await _release_inference(inference, abandoned=gpu is not None and gpu.abandoned_inference)

        stack.push_async_callback(release)
        clock = SystemClock()
        engine, uow, store = await _storage(settings, stack)
        workspace = JobWorkspace(settings.workspace_root)
        publisher = AssetPublisher(store)
        listener = PostgresListeningJobQueue(
            connect=_notify_connect(settings), local=AsyncioJobQueue(maxsize=settings.jobs.queue_size)
        )
        waiters = AsyncioJobWaiters()
        batches = BatchService(uow=uow, publisher=publisher, workspace=workspace, waiters=waiters, clock=clock)
        worker_id = lease_owner()
        handlers = _job_handlers(
            settings,
            model_host=inference.model_host,
            uow=uow,
            store=store,
            media=FfmpegMediaTool.from_path(),
            publisher=publisher,
            workspace=workspace,
        )
        gpu = GpuWorker(
            worker_id=worker_id,
            version=APP_VERSION,
            lock=PostgresWorkerLock(engine),
            worker=_job_worker(
                settings,
                uow=uow,
                queue=listener,
                waiters=waiters,
                handlers=handlers,
                batches=batches,
                workspace=workspace,
                clock=clock,
                lease_owner=worker_id,
            ),
            recovery=_recovery(settings, uow=uow, queue=listener, batches=batches, clock=clock),
            model_host=inference.model_host,
            runtime=inference.runtime,
            record_preload_errors=inference.runtime.record_preload_errors,
            uow=uow,
            clock=clock,
            monotonic=SystemMonotonicClock(),
            heartbeat_seconds=settings.jobs.heartbeat_seconds,
            poll_interval_seconds=settings.jobs.poll_interval_seconds,
            drain_timeout_seconds=settings.gpu_worker.drain_timeout_seconds,
            startup_recovery=settings.jobs.startup_recovery,
            preload=tuple(dict.fromkeys((*_window_preload(settings, clock.now()), *preload))),
            loops=(listener.run, *_idle_loops(settings, inference.model_host)),
        )
        yield gpu


async def run_gpu_worker(settings: Settings, *, preload: tuple[ModelRef, ...], register_only: bool) -> bool:
    """Run the worker role, or only publish its capabilities; return whether a native call was abandoned."""
    async with gpu_worker_session(settings, preload=preload) as gpu:
        if register_only:
            await gpu.register_only()
            return False
        loop = asyncio.get_running_loop()
        for signal_number in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signal_number, gpu.request_stop)
        try:
            await gpu.run()
        finally:
            for signal_number in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(signal_number)
        return gpu.abandoned_inference


def run_worker(settings: Settings, *, preload: tuple[ModelRef, ...] = (), register_only: bool = False) -> None:
    """Run the worker role; after an abandoned native call use os._exit, as the interpreter would join its thread."""
    configure_logging(level=settings.observability.log_level, log_format=settings.observability.log_format)
    if asyncio.run(run_gpu_worker(settings, preload=preload, register_only=register_only)):
        logging.shutdown()
        os._exit(0)
