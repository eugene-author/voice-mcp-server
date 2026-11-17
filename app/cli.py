"""Command-line entry points: serve, diagnose, migrate, models, jobs, voices and eval."""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn

import typer
from alembic import command
from alembic.config import Config

from app.adapters.host.probe import SystemHostProbe
from app.adapters.media.ffmpeg.tool import FfmpegMediaTool
from app.adapters.models.factory import build_model_catalog
from app.core.clock import SystemClock
from app.core.logging import configure_logging
from app.core.settings import Settings, load_settings
from app.db.engine import create_engine, create_session_factory
from app.db.provisioning import ensure_database
from app.db.unit_of_work import SqlUnitOfWorkFactory
from app.domain.errors import VoiceMcpError
from app.domain.ids import JobId, VoiceId
from app.domain.jobs import JobStatus
from app.domain.models import ModelKind
from app.domain.workers import ModelRef
from app.main import (
    build_inference,
    diagnostics_session,
    enforce_offline_runtime,
    gpu_operator_session,
    prepare_data_root,
    run,
    run_worker,
    running_container,
)
from app.services.evaluation import evaluate_dataset
from app.services.runtime_selection import select_stt_backend, select_tts_backend
from app.workers.job_transitions import JobTransitions

app = typer.Typer(help="Voice MCP Server: local speech recognition, synthesis and voice cloning.", no_args_is_help=True)
models_app = typer.Typer(help="Install and list models.", no_args_is_help=True)
jobs_app = typer.Typer(help="Inspect, retry and cancel jobs.", no_args_is_help=True)
voices_app = typer.Typer(help="Manage cloned voices.", no_args_is_help=True)
eval_app = typer.Typer(help="Offline evaluation.", no_args_is_help=True)
gpu_app = typer.Typer(help="Inspect, start, drain and stop the GPU worker's compute.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(jobs_app, name="jobs")
app.add_typer(voices_app, name="voices")
app.add_typer(eval_app, name="eval")
app.add_typer(gpu_app, name="gpu")

TransportOption = Annotated[
    Literal["streamable_http", "stdio"] | None, typer.Option("--transport", help="Override server.transport.")
]
RoleOption = Annotated[
    Literal["all", "api", "worker"] | None,
    typer.Option("--role", help="Override deployment.role: all (default) or api; use the worker command for workers."),
]


def _settings(transport: str | None = None, role: str | None = None) -> Settings:
    """Load settings, applying transport and role overrides and configuring logging."""
    overrides: dict[str, Any] = {"server": {"transport": transport}} if transport else {}
    if role:
        overrides["deployment"] = {"role": role}
    settings = load_settings(**overrides)
    configure_logging(level=settings.observability.log_level, log_format=settings.observability.log_format)
    return settings


def _echo_json(data: Any) -> None:
    """Print data as indented JSON."""
    typer.echo(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def _fail(message: str) -> NoReturn:
    """Print an error and exit non-zero."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def serve(transport: TransportOption = None, role: RoleOption = None) -> None:
    """Run the MCP server (all: with the in-process worker; api: without models); errors exit non-zero."""
    settings = _settings(transport, role)
    if settings.deployment.role == "worker":
        _fail("serve runs the all or api role; start a GPU worker with: python -m app worker")
    try:
        run(settings)
    except* VoiceMcpError as group:
        _fail("; ".join(exc.message for exc in group.exceptions if isinstance(exc, VoiceMcpError)))


@app.command()
def worker(
    register_only: Annotated[
        bool, typer.Option(help="Publish this host's capabilities without loading models, then exit.")
    ] = False,
    preload: Annotated[
        list[str] | None, typer.Option(help="Load kind:name at startup, e.g. stt:large-v3; repeatable.")
    ] = None,
) -> None:
    """Run the GPU worker role: claim and run jobs, drain on SIGTERM or a controller request, exit 0 when stopped."""
    try:
        refs = tuple(ModelRef.parse(entry) for entry in preload or ())
    except ValueError as exc:
        _fail(str(exc))
    settings = _settings("stdio", "worker")
    try:
        run_worker(settings, preload=refs, register_only=register_only)
    except* VoiceMcpError as group:
        _fail("; ".join(exc.message for exc in group.exceptions if isinstance(exc, VoiceMcpError)))


@app.command()
def migrate(
    revision: Annotated[str, typer.Argument(help="Target revision.")] = "head",
    create_database: Annotated[
        bool, typer.Option("--create-database", help="Create the database in DATABASE_URL if it is missing.")
    ] = False,
) -> None:
    """Apply Alembic migrations to DATABASE_URL; with --create-database, create a missing database first."""
    settings = _settings("stdio")
    try:
        created = asyncio.run(ensure_database(settings.database, create=create_database))
    except VoiceMcpError as exc:
        _fail(exc.message)
    if created:
        typer.echo("created the database in DATABASE_URL")
    config = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    command.upgrade(config, revision)
    typer.echo(f"database upgraded to {revision}")


@app.command()
def diagnose(
    skip_inference: Annotated[
        bool, typer.Option(help="Only report host, models and dependencies; load no weights.")
    ] = False,
) -> None:
    """Report host profile, backend selection, installed models, database and a one-second STT probe."""
    settings = _settings("stdio")
    try:
        report = asyncio.run(_diagnose(settings, include_inference=not skip_inference))
    except VoiceMcpError as exc:
        _fail(exc.message)
    _echo_json(report)


async def _diagnose(settings: Settings, *, include_inference: bool) -> dict[str, Any]:
    """Run the diagnostics inside a session that always releases models, the executor and the database engine."""
    async with diagnostics_session(settings) as health:
        report = await health.run_diagnostics(include_inference=include_inference)
    return report.to_dict()


@models_app.command("install")
def models_install(
    stt: Annotated[str | None, typer.Option(help="Whisper model name, e.g. small.")] = None,
    tts: Annotated[str | None, typer.Option(help="Qwen3-TTS checkpoint, e.g. Qwen3-TTS-12Hz-0.6B-Base.")] = None,
    denoise: Annotated[str | None, typer.Option(help="Denoise model, e.g. deepfilternet3.")] = None,
    backend: Annotated[str, typer.Option(help="auto, ctranslate2 or mlx (STT); auto, torch or mlx (TTS).")] = "auto",
    source: Annotated[
        Path | None, typer.Option("--from", help="Copy weights from a local directory instead of downloading.")
    ] = None,
) -> None:
    """Install one model; this is the only command that downloads weights."""
    settings = _settings("stdio")
    prepare_data_root(settings)
    chosen = [
        (kind, name)
        for kind, name in ((ModelKind.STT, stt), (ModelKind.TTS, tts), (ModelKind.DENOISE, denoise))
        if name
    ]
    if len(chosen) != 1:
        _fail("pass exactly one of --stt, --tts or --denoise")
    kind, name = chosen[0]
    resolved_backend = _install_backend(settings, kind, backend)
    catalog = build_model_catalog(settings)
    try:
        installed = catalog.install(kind, resolved_backend, str(name), source_dir=source.resolve() if source else None)
    except VoiceMcpError as exc:
        _fail(exc.message)
    typer.echo(f"installed {kind}/{installed.backend}/{installed.name} -> {installed.local_path}")


def _install_backend(settings: Settings, kind: ModelKind, backend: str) -> str:
    """Resolve the backend an install targets, using the host's zero-config selection for auto."""
    if kind is ModelKind.DENOISE:
        return "deepfilternet"
    if backend != "auto":
        return backend
    host = SystemHostProbe().probe()
    if kind is ModelKind.STT:
        return str(select_stt_backend(settings.stt.backend, host))
    selected = select_tts_backend(settings.tts.backend, host)
    return str(selected) if selected else "torch"


@models_app.command("list")
def models_list(kind: Annotated[str | None, typer.Option(help="stt, tts or denoise.")] = None) -> None:
    """List registry models and whether they are installed."""
    settings = _settings("stdio")
    catalog = build_model_catalog(settings)
    installed = {(m.kind, m.backend, m.name) for m in catalog.installed()}
    for entry in catalog.entries(ModelKind(kind) if kind else None):
        mark = "installed" if (entry.kind, entry.backend, entry.name) in installed else "-"
        typer.echo(f"{entry.kind:<8} {entry.backend:<12} {entry.name:<34} {entry.approx_size_mb:>6} MB  {mark}")


def _job_id(raw: str) -> JobId:
    """Parse a job ID or exit."""
    try:
        return JobId(uuid.UUID(raw))
    except ValueError:
        _fail("job_id must be a UUID")


@jobs_app.command("show")
def jobs_show(job_id: str) -> None:
    """Show a job's status, phase, error and files."""
    asyncio.run(_jobs_show(_job_id(job_id), _settings("stdio")))


async def _jobs_show(job_id: JobId, settings: Settings) -> None:
    """Print one job without an ownership check (trusted CLI)."""
    async with _database(settings) as uow_factory, uow_factory() as uow:
        view = await uow.jobs.get(job_id, None)
    if view is None:
        _fail("job not found")
    _echo_json(
        {
            "job_id": str(view.id),
            "type": str(view.type),
            "status": str(view.status),
            "progress": view.progress,
            "phase": view.phase,
            "error": {"code": view.error.code, "message": view.error.message} if view.error else None,
            "warnings": list(view.warnings),
            "files": view.files,
            "counts": view.counts.to_dict() if view.counts else None,
        }
    )


@jobs_app.command("result")
def jobs_result(job_id: str) -> None:
    """Print a finished job's result payload and files."""
    asyncio.run(_jobs_result(_job_id(job_id), _settings("stdio")))


async def _jobs_result(job_id: JobId, settings: Settings) -> None:
    """Print the result of one job."""
    async with _database(settings) as uow_factory, uow_factory() as uow:
        view = await uow.jobs.get(job_id, None)
    if view is None:
        _fail("job not found")
    _echo_json({"status": str(view.status), "result": view.result, "files": view.files})


@jobs_app.command("retry")
def jobs_retry(job_id: str) -> None:
    """Requeue a failed job with one more attempt; the running server picks it up within one poll interval."""
    asyncio.run(_jobs_transition(_job_id(job_id), JobStatus.QUEUED, _settings("stdio")))


@jobs_app.command("cancel")
def jobs_cancel(job_id: str) -> None:
    """Cancel a queued or retry-waiting job; running jobs cannot be interrupted."""
    asyncio.run(_jobs_transition(_job_id(job_id), JobStatus.CANCELLED, _settings("stdio")))


async def _jobs_transition(job_id: JobId, status: JobStatus, settings: Settings) -> None:
    """Apply a manual transition through job transitions so registration voices stay consistent."""
    async with _database(settings) as uow_factory:
        transitions = JobTransitions(
            uow=uow_factory,
            clock=SystemClock(),
            retry_backoff_seconds=settings.jobs.retry_backoff_seconds,
            save_events=settings.observability.save_job_events,
        )
        changed = await (transitions.retry(job_id) if status is JobStatus.QUEUED else transitions.cancel(job_id))
    if not changed:
        _fail(f"job cannot move to {status} from its current state (or its voice was deleted)")
    typer.echo(f"job {job_id} -> {status}")


@voices_app.command("delete")
def voices_delete(voice_id: str) -> None:
    """Delete a voice and purge every stored copy of its reference audio."""
    asyncio.run(_voices_delete(VoiceId(voice_id), _settings("stdio")))


async def _voices_delete(voice_id: VoiceId, settings: Settings) -> None:
    """Purge one voice using the serving container's voice service."""
    async with running_container(settings, transport="stdio") as container:
        try:
            removed = await container.voices.delete_voice(voice_id, owner=None)
        except VoiceMcpError as exc:
            _fail(exc.message)
    typer.echo(f"voice {voice_id} deleted ({removed} stored assets removed)")


@eval_app.command("wer")
def eval_wer(
    dataset: Annotated[Path, typer.Option(help="Directory of audio files with matching .txt ground truth.")],
    model: Annotated[str | None, typer.Option(help="Whisper model name.")] = None,
    language: Annotated[str | None, typer.Option(help="ISO language code.")] = None,
) -> None:
    """Transcribe a ground-truth dataset offline and report WER and CER per file and overall."""
    settings = _settings("stdio")
    enforce_offline_runtime()
    inference = build_inference(settings)
    try:
        report = asyncio.run(
            evaluate_dataset(
                dataset.resolve(),
                model_host=inference.model_host,
                converter=FfmpegMediaTool.from_path(),
                model=model or settings.stt.model,
                language=language,
                workspace=settings.workspace_root,
            )
        )
    except VoiceMcpError as exc:
        _fail(exc.message)
    finally:
        inference.executor.shutdown()
    _echo_json(report)


CONTROLLER_WARNING = "warning: the wake controller is enabled and may reverse this action on its next tick"


def _gpu(action: str, *, force: bool = False) -> None:
    """Run one gpu action, warning first when the wake controller could reverse it."""
    settings = _settings("stdio")
    if action != "status" and settings.gpu_worker.on_demand.enabled:
        typer.secho(CONTROLLER_WARNING, fg=typer.colors.YELLOW, err=True)
    try:
        asyncio.run(_gpu_action(settings, action, force=force))
    except VoiceMcpError as exc:
        _fail(exc.message)


async def _gpu_action(settings: Settings, action: str, *, force: bool) -> None:
    """Perform one operator action and print its result."""
    async with gpu_operator_session(settings) as operator:
        match action:
            case "status":
                _echo_json(await operator.status())
            case "start":
                await operator.start()
                typer.echo("start requested")
            case "drain":
                typer.echo(f"drain requested for worker {await operator.drain()}")
            case "stop":
                await operator.stop(force=force)
                typer.echo("stop requested" + (" (forced)" if force else ""))


@gpu_app.command("status")
def gpu_status() -> None:
    """Print the target state, job demand, what the controller would decide now, and the worker rows."""
    _gpu("status")


@gpu_app.command("start")
def gpu_start() -> None:
    """Start the GPU worker's compute once."""
    _gpu("start")


@gpu_app.command("drain")
def gpu_drain() -> None:
    """Ask the active worker to finish its current job and stop."""
    _gpu("drain")


@gpu_app.command("stop")
def gpu_stop(
    force: Annotated[bool, typer.Option(help="Stop even while a job is running (it is released or retried).")] = False,
) -> None:
    """Stop the GPU worker's compute; refused while a job runs unless --force."""
    _gpu("stop", force=force)


@asynccontextmanager
async def _database(settings: Settings) -> AsyncIterator[SqlUnitOfWorkFactory]:
    """Yield a unit-of-work factory over a short-lived engine for CLI commands."""
    engine = create_engine(settings.database)
    try:
        yield SqlUnitOfWorkFactory(create_session_factory(engine))
    finally:
        await engine.dispose()


logging.captureWarnings(True)
