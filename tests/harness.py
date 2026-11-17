"""Assembles real services and workers on top of port mocks, mirroring app.main."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.adapters.queue.asyncio.queue import AsyncioJobWaiters
from app.core.executor import InferenceRunner
from app.core.settings import Settings, load_presets, load_settings
from app.domain.caller import Caller
from app.domain.jobs import JobType
from app.domain.models import ModelKind
from app.domain.workers import CapabilitySnapshot, WorkerRecord, WorkerStatus
from app.main import InferenceComponents
from app.mcp_server.auth import CallerResolver
from app.mcp_server.dependencies import ToolServices
from app.ports.runtime import RuntimeCatalog
from app.services.batch_service import BatchService
from app.services.health_check_service import HealthCheckService
from app.services.inference_progress import DEFAULT_INTERVAL_SECONDS, InferenceProgress
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
from app.services.staged_job_assets import StagedJobAssets
from app.services.synthesis_service import SynthesisService
from app.services.transcription_options_resolver import TranscriptionOptionsResolver
from app.services.transcription_service import TranscriptionService
from app.services.voice_registration_service import VoiceRegistrationService
from app.services.voice_service import VoiceService
from app.workers.heartbeat import Heartbeat
from app.workers.job_transitions import JobTransitions
from app.workers.job_worker import JobHandler, JobWorker
from app.workers.recovery import JobRecovery
from tests.mocks.clock import MockClock, MockMonotonicClock
from tests.mocks.denoise import MockDenoiser
from tests.mocks.host import host_profile
from tests.mocks.media import MockMedia
from tests.mocks.models import MockModelCatalog
from tests.mocks.queue import MockJobQueue
from tests.mocks.storage import MockAssetStore
from tests.mocks.stt import MockSttBackend
from tests.mocks.tts import MockTtsEngine
from tests.mocks.unit_of_work import MockUnitOfWorkFactory
from tests.mocks.vad import MockVoiceActivityDetector

OWNER = Caller("https://idp.example/realms/voice", "user-1", "agent")
OTHER = Caller("https://idp.example/realms/voice", "user-2", "agent")
DEFAULT_INSTALLED = {
    (ModelKind.STT, "ctranslate2", "small"),
    (ModelKind.STT, "mlx", "small"),
    (ModelKind.TTS, "torch", "Qwen3-TTS-12Hz-0.6B-Base"),
    (ModelKind.TTS, "torch", "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
    (ModelKind.DENOISE, "deepfilternet", "deepfilternet3"),
}


@dataclass
class Harness:
    """Every component of a test application."""

    settings: Settings
    inputs: Path
    stt: MockSttBackend
    tts: MockTtsEngine | None
    denoiser: MockDenoiser | None
    media: MockMedia
    store: MockAssetStore
    uow: MockUnitOfWorkFactory
    queue: MockJobQueue
    waiters: AsyncioJobWaiters
    clock: MockClock
    monotonic: MockMonotonicClock
    model_host: ModelHost
    runtime: RuntimeCatalog
    """What the services validate against: the local runtime, or the registry snapshot in the api role."""

    local_runtime: LocalRuntimeCatalog
    planner: JobPlanner
    queries: JobQueryService
    jobs: JobService
    voices: VoiceService
    batches: BatchService
    transitions: JobTransitions
    worker: JobWorker
    recovery: JobRecovery
    stt_options: TranscriptionOptionsResolver
    staged: StagedJobAssets
    handlers: dict[JobType, JobHandler]
    stt_progress: InferenceProgress
    workspace: JobWorkspace
    executor: InferenceRunner
    extra: dict[str, Any] = field(default_factory=dict)

    def tool_services(self, caller: Caller = OWNER) -> ToolServices:
        """Return tool services whose caller resolver always yields caller."""
        return ToolServices(
            caller_resolver=CallerResolver(trusted_local=caller),
            jobs=self.jobs,
            queries=self.queries,
            models=ModelService(catalog=self.model_host._catalog, runtime=self.runtime),
            voices=self.voices,
            health=HealthCheckService(
                host=host_profile(),
                runtime=self.runtime,
                catalog=self.model_host._catalog,
                checks={"storage": self.store.verify},
                worker_state=lambda: "active",
                data_root=self.settings.data_root,
                workspace_root=self.settings.workspace_root,
                model_host=self.model_host,
            ),
            inline_wait_seconds=0.5,
        )

    async def run_queued(self) -> None:
        """Process every notified job until the queue is empty."""
        while self.queue.notified:
            job_id = self.queue.notified.pop(0)
            await self.queue.get()
            await self.worker._process(job_id)
            self.queue.task_done()


async def publish_worker(
    uow: MockUnitOfWorkFactory,
    snapshot: CapabilitySnapshot,
    *,
    at: datetime,
    worker_id: str = "gpu-host:1",
    status: WorkerStatus = WorkerStatus.READY,
) -> WorkerRecord:
    """Register a worker row with a snapshot, as a GPU worker does at startup."""
    record = WorkerRecord(
        id=worker_id, status=status, capabilities=snapshot, started_at=at, heartbeat_at=at, version="test"
    )
    async with uow() as work:
        await work.workers.register(record)
    return record


def build_harness(
    tmp_path: Path,
    *,
    stt: MockSttBackend | None = None,
    tts: MockTtsEngine | bool | None = True,
    denoiser: bool = False,
    vad: MockVoiceActivityDetector | None = None,
    installed: set[tuple[ModelKind, str, str]] | None = None,
    store: MockAssetStore | None = None,
    media: MockMedia | None = None,
    settings_overrides: dict[str, Any] | None = None,
    progress_interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    inference: InferenceComponents | None = None,
    registry: bool = False,
    uow: Any = None,
    queue: Any = None,
) -> Harness:
    """Build the test application in tmp_path; real inference replaces the mocks, registry=True mirrors the api role."""
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, Any] = {
        "server": {"transport": "stdio", "allowed_input_directories": [str(inputs)]},
        "storage": {"data_root": str(tmp_path / "data")},
        "jobs": {"retry_backoff_seconds": [1, 2]},
    }
    for key, value in (settings_overrides or {}).items():
        overrides[key] = {**overrides.get(key, {}), **value}
    settings = load_settings(**overrides)
    if inference is None:
        stt_backend = stt or MockSttBackend()
        tts_engine = MockTtsEngine() if tts is True else (tts or None)
        denoise = MockDenoiser() if denoiser else None
        executor = InferenceRunner()
        catalog = MockModelCatalog(installed=DEFAULT_INSTALLED if installed is None else installed)
        model_host = ModelHost(
            stt=stt_backend,
            tts=tts_engine,
            denoiser=denoise,
            vad=vad,
            catalog=catalog,
            executor=executor,
            exclusive=False,
        )
        local_runtime = LocalRuntimeCatalog(
            model_host=model_host,
            catalog=catalog,
            profile="linux-cpu",
            denoise_device=denoise.device if denoise else None,
        )
    else:
        stt_backend, tts_engine, denoise = inference.stt, inference.tts, inference.denoiser
        executor, model_host, local_runtime = inference.executor, inference.model_host, inference.runtime
    clock, monotonic = MockClock(), MockMonotonicClock()
    uow = uow or MockUnitOfWorkFactory()
    queue = queue or MockJobQueue()
    waiters = AsyncioJobWaiters()
    runtime: RuntimeCatalog = RegistryRuntimeCatalog(uow=uow, refresh_seconds=15) if registry else local_runtime
    asset_store = store or MockAssetStore()
    media_tool = media or MockMedia()
    workspace = JobWorkspace(tmp_path / "workspace")
    publisher = AssetPublisher(asset_store)
    policy = InputPathPolicy(
        allowed_directories=[inputs],
        accepted_formats=settings.audio.accepted_formats,
        max_input_bytes=settings.audio.max_input_bytes,
    )
    presets = load_presets()
    stt_options = TranscriptionOptionsResolver(settings=settings, presets=presets, runtime=runtime)
    planner = JobPlanner(
        settings=settings,
        presets=presets,
        stt_options=stt_options,
        runtime=runtime,
        input_policy=policy,
        uow=uow,
        clock=clock,
    )
    queries = JobQueryService(
        uow=uow, store=asset_store, default_limit=20, max_limit=100, presign_enabled=True, presign_ttl_seconds=300
    )
    jobs = JobService(
        uow=uow,
        planner=planner,
        queue=queue,
        waiters=waiters,
        queries=queries,
        runtime=runtime,
        clock=clock,
        monotonic=monotonic,
        uploader=(
            InputUploader(store=asset_store, max_batch_bytes=settings.audio.max_batch_upload_bytes)
            if settings.storage.input_transfer == "ingest"
            else None
        ),
    )
    voices = VoiceService(
        uow=uow,
        store=asset_store,
        runtime=runtime,
        preset_checkpoint=settings.tts.preset_checkpoint,
        clock=clock,
    )
    staged = StagedJobAssets(workspace)
    stt_progress = InferenceProgress(
        monotonic=monotonic, seconds_per_unit=1.0, interval_seconds=progress_interval_seconds
    )
    tts_progress = InferenceProgress(
        monotonic=monotonic, seconds_per_unit=1.0, interval_seconds=progress_interval_seconds
    )
    handlers: dict[JobType, JobHandler] = {
        JobType.SPEECH_TO_TEXT: TranscriptionService(
            probe=media_tool,
            converter=media_tool,
            model_host=model_host,
            publisher=publisher,
            staged=staged,
            workspace=workspace,
            fillers=settings.audio.fillers,
            max_duration_seconds=settings.audio.max_duration_seconds,
            text_preview_chars=settings.jobs.text_preview_chars,
            inference_progress=stt_progress,
        ),
        JobType.TEXT_TO_SPEECH: SynthesisService(
            uow=uow,
            store=asset_store,
            converter=media_tool,
            model_host=model_host,
            publisher=publisher,
            staged=staged,
            workspace=workspace,
            cloning_checkpoint=settings.tts.cloning_checkpoint,
            preset_checkpoint=settings.tts.preset_checkpoint,
            inference_progress=tts_progress,
        ),
        JobType.REGISTER_VOICE: VoiceRegistrationService(
            probe=media_tool,
            converter=media_tool,
            model_host=model_host,
            publisher=publisher,
            workspace=workspace,
            cloning_checkpoint=settings.tts.cloning_checkpoint,
            verification_stt_model=settings.stt.model,
            inference_progress=tts_progress,
        ),
    }
    batches = BatchService(uow=uow, publisher=publisher, workspace=workspace, waiters=waiters, clock=clock)
    transitions = JobTransitions(uow=uow, clock=clock, retry_backoff_seconds=(1, 2), save_events=True)
    worker = JobWorker(
        transitions=transitions,
        handlers=handlers,
        queue=queue,
        waiters=waiters,
        batches=batches,
        workspace=workspace,
        heartbeat=Heartbeat(uow=uow, clock=clock, interval_seconds=60),
        lease_owner="test-worker",
    )
    recovery = JobRecovery(
        uow=uow, queue=queue, batches=batches, clock=clock, lease_timeout_seconds=120, poll_interval_seconds=60
    )
    return Harness(
        settings=settings,
        inputs=inputs,
        stt=stt_backend,
        tts=tts_engine,
        denoiser=denoise,
        media=media_tool,
        store=asset_store,
        uow=uow,
        queue=queue,
        waiters=waiters,
        clock=clock,
        monotonic=monotonic,
        model_host=model_host,
        runtime=runtime,
        local_runtime=local_runtime,
        planner=planner,
        queries=queries,
        jobs=jobs,
        voices=voices,
        batches=batches,
        transitions=transitions,
        worker=worker,
        recovery=recovery,
        stt_options=stt_options,
        staged=staged,
        handlers=handlers,
        stt_progress=stt_progress,
        workspace=workspace,
        executor=executor,
    )
