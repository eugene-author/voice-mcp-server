"""Typed settings; precedence: init arguments, environment, short aliases, .env, then default.yaml."""

import ipaddress
import os
from collections.abc import Mapping
from datetime import time, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from app.domain.errors import InvalidRequestError
from app.domain.stt import STT_MODEL_NAMES, SttBackendChoice, SttOptions, VadSettings
from app.domain.tts import OUTPUT_FORMATS, TtsBackendChoice, TtsOptions
from app.domain.workers import WEEKDAYS, ModelRef, Schedule, ScheduleWindow

ALLOWED_JWT_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512"})

RETIRED_ENV_ALIASES = {"SCRATCH_ROOT": "WORKSPACE_ROOT"}
"""Short variables that were renamed; setting one fails startup instead of being ignored."""

SHORT_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "WHISPER_BACKEND": ("stt", "backend"),
    "TTS_BACKEND": ("tts", "backend"),
    "DATABASE_URL": ("database", "url"),
    "STORAGE_BACKEND": ("storage", "backend"),
    "DATA_ROOT": ("storage", "data_root"),
    "WORKSPACE_ROOT": ("storage", "workspace_root"),
    "ASSETS_ROOT": ("storage", "local", "assets_root"),
    "MODELS_ROOT": ("models", "root"),
    "SERVICE_ROLE": ("deployment", "role"),
    "GPU_ON_DEMAND": ("gpu_worker", "on_demand", "enabled"),
}
"""Short environment variables mapped to nested settings paths."""


class _Section(BaseModel):
    """Base for settings sections: immutable and strict about unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ServerSettings(_Section):
    """MCP server identity, transport and input allowlist."""

    name: str = "local-voice-mcp"
    transport: Literal["streamable_http", "stdio"] = "streamable_http"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    allowed_input_directories: tuple[Path, ...] = ()
    """Empty means ${data_root}/inputs."""

    allowed_hosts: tuple[str, ...] = ()
    """Host header values accepted over HTTP (host:port, or host:* for any port); empty means loopback only."""

    allowed_origins: tuple[str, ...] = ()
    """Origin header values accepted over HTTP; empty means loopback origins only; requests without Origin pass."""

    @property
    def binds_loopback(self) -> bool:
        """Return whether the HTTP listener is reachable from this machine only."""
        try:
            return ipaddress.ip_address(self.host.strip("[]")).is_loopback
        except ValueError:
            return self.host == "localhost"

    @model_validator(mode="after")
    def _check_allowed_hosts(self) -> Self:
        """Require explicit allowed_hosts when Streamable HTTP listens beyond loopback."""
        if self.transport == "streamable_http" and not self.binds_loopback and not self.allowed_hosts:
            raise ValueError(
                f"server.host={self.host} is not loopback, so server.allowed_hosts must list the Host headers clients "
                "send (for example voice.example.com or 127.0.0.1:8000)"
            )
        return self


class AuthConfig(_Section):
    """OAuth resource-server settings."""

    mode: Literal["oauth", "trusted_local"] = "oauth"
    issuer_url: str | None = None
    audience: str | None = None
    resource_server_url: str | None = None
    validate_token_resource: bool = True
    jwks_url: str | None = None
    """Where the signing keys are fetched; set it when the issuer's public URL is not reachable from the server
    (for example http://keycloak:8080/... inside Compose). Issuer validation always uses issuer_url."""
    algorithms: tuple[str, ...] = ("RS256", "ES256")
    required_scopes: tuple[str, ...] = ("mcp:access",)
    leeway_seconds: int = Field(default=30, ge=0)
    jwks_cache_seconds: int = Field(default=3600, ge=1)
    jwks_min_refresh_seconds: int = Field(default=30, ge=0)
    jwks_fetch_timeout_seconds: float = Field(default=10, gt=0)
    warm_jwks_on_startup: bool = True
    startup_timeout_seconds: float = Field(default=60, ge=0)
    """How long startup keeps retrying OIDC discovery and the first key fetch before failing."""

    @field_validator("jwks_url", "issuer_url", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat an empty string (an unset Compose variable) as not configured."""
        return None if value == "" else value

    @model_validator(mode="after")
    def _check_algorithms(self) -> Self:
        """Reject none, HMAC and unknown signature algorithms."""
        unsupported = set(self.algorithms) - ALLOWED_JWT_ALGORITHMS
        if unsupported or not self.algorithms:
            raise ValueError(f"unsupported JWT algorithms: {sorted(unsupported)}")
        return self


class HardwareSettings(_Section):
    """Memory policy for loaded models."""

    exclusive_residency_ram_mb: int = 16384


class VadConfig(_Section):
    """Silero VAD parameters."""

    enabled: bool = True
    threshold: float = Field(default=0.5, gt=0, lt=1)
    min_speech_duration_ms: int = Field(default=250, ge=0)
    min_silence_duration_ms: int = Field(default=500, ge=0)
    speech_pad_ms: int = Field(default=250, ge=0)
    merge_gap_ms: int = Field(default=300, ge=0)

    def to_domain(self) -> VadSettings:
        """Return the domain value."""
        return VadSettings(**self.model_dump())


class SttSettings(_Section):
    """Speech-to-text defaults."""

    backend: SttBackendChoice = SttBackendChoice.AUTO
    model: str = "small"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "auto"
    low_vram_threshold_mb: int = 4000
    language: str = "auto"
    beam_size: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.0, ge=0, le=1)
    word_timestamps: bool = True
    segment_flags: bool = True
    glossary: tuple[str, ...] = ()
    vad: VadConfig = VadConfig()

    @model_validator(mode="after")
    def _check_model(self) -> Self:
        """Reject unknown Whisper model sizes."""
        if self.model not in STT_MODEL_NAMES:
            raise ValueError(f"stt.model must be one of {STT_MODEL_NAMES}")
        return self


class TtsSettings(_Section):
    """Text-to-speech defaults."""

    backend: TtsBackendChoice = TtsBackendChoice.AUTO
    engine: Literal["qwen3_tts"] = "qwen3_tts"
    cloning_checkpoint: str = "Qwen3-TTS-12Hz-0.6B-Base"
    preset_checkpoint: str | None = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
    output_format: str = "wav"
    allow_cpu: bool = True
    conditioning_cache_size: int = Field(default=8, ge=1)
    max_characters: int = Field(default=20000, ge=1)
    max_chunk_chars: int = Field(default=300, ge=50)
    gap_ms: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def _check_format(self) -> Self:
        """Reject unknown output formats."""
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(f"tts.output_format must be one of {OUTPUT_FORMATS}")
        return self


class DenoisingSettings(_Section):
    """DeepFilterNet settings."""

    engine: Literal["deepfilternet"] = "deepfilternet"
    model: str = "deepfilternet3"
    enabled: bool = False
    attenuation_limit_db: float = Field(default=0.0, ge=0, le=100)
    """0 removes all noise; larger values keep up to this much attenuation for a more natural sound."""


class AudioSettings(_Section):
    """Input limits and audio processing defaults."""

    accepted_formats: tuple[str, ...] = ("mp3", "wav", "m4a", "ogg", "flac", "opus", "webm")
    max_input_bytes: int = 2 * 1024**3
    max_duration_seconds: float = 14400
    denoising: DenoisingSettings = DenoisingSettings()
    loudness_normalization: bool = True
    filler_removal: Literal["annotate_only", "remove"] = "annotate_only"
    repeated_starts: Literal["annotate_only", "remove"] = "annotate_only"
    fillers: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    max_batch_upload_bytes: int = Field(default=10 * 1024**3, ge=1)
    """Total input size a transcribe_batch may copy into the asset store with storage.input_transfer=ingest."""


class JobsSettings(_Section):
    """Worker, retry and listing settings."""

    max_attempts: int = Field(default=2, ge=1)
    retry_backoff_seconds: tuple[float, ...] = (3, 10)
    startup_recovery: bool = True
    poll_interval_seconds: float = Field(default=5, gt=0)
    heartbeat_seconds: float = Field(default=15, gt=0)
    lease_timeout_seconds: float = Field(default=120, gt=0)
    inline_wait_seconds: float = Field(default=120, ge=0)
    batch_max_items: int = Field(default=500, ge=1)
    text_preview_chars: int = Field(default=2000, ge=0)
    list_jobs_default_limit: int = Field(default=20, ge=1)
    list_jobs_max_limit: int = Field(default=100, ge=1)
    queue_size: int = Field(default=1000, ge=1)


class LocalStorageSettings(_Section):
    """Local filesystem asset storage."""

    assets_root: Path | None = None


class S3Settings(_Section):
    """Amazon S3 asset storage; credentials come from the boto3 provider chain only."""

    bucket: str | None = None
    region: str | None = None
    prefix: str = "voice-mcp"
    presign_downloads: bool = False
    presign_ttl_seconds: int = Field(default=300, ge=1, le=3600)
    encryption: Literal["sse-s3", "sse-kms"] = "sse-s3"
    kms_key_id: str | None = None
    max_concurrency: int = Field(default=4, ge=1)


class StorageSettings(_Section):
    """Asset storage backend and runtime data locations."""

    backend: Literal["local", "s3"] = "local"
    data_root: Path = Path("./var")
    preserve_inputs: bool = True
    input_transfer: Literal["shared_path", "ingest"] = "shared_path"
    """shared_path: workers read inputs at the submitted path; ingest: inputs are copied into the asset store."""

    orphan_retention_hours: float = Field(default=24, ge=1)
    """Unreferenced stored assets older than this are deleted by reconciliation."""

    workspace_root: Path | None = None
    local: LocalStorageSettings = LocalStorageSettings()
    s3: S3Settings = S3Settings()


class DeploymentSettings(_Section):
    """Which part of the system this process runs."""

    role: Literal["all", "api", "worker"] = "all"
    """all: one process (MCP server and inference); api: MCP server without models; worker: GPU inference only."""


class ScheduleWindowSettings(_Section):
    """A daily warm window for the GPU worker."""

    days: tuple[Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"], ...] = Field(min_length=1)
    start: time
    end: time
    lead_minutes: int = Field(default=0, ge=0)
    preload: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        """Require start before end (no window spans midnight) and kind:name preload entries."""
        if self.start >= self.end:
            raise ValueError(f"schedule window start {self.start} must be before end {self.end}")
        for entry in self.preload:
            ModelRef.parse(entry)
        return self

    def to_domain(self) -> ScheduleWindow:
        """Return the domain window."""
        return ScheduleWindow(
            days=frozenset(WEEKDAYS.index(day) for day in self.days),
            start=self.start,
            end=self.end,
            lead=timedelta(minutes=self.lead_minutes),
            preload=tuple(ModelRef.parse(entry) for entry in self.preload),
        )


class ScheduleSettings(_Section):
    """Warm windows and the timezone they are written in."""

    timezone: str = "UTC"
    windows: tuple[ScheduleWindowSettings, ...] = ()

    @field_validator("timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        """Require an IANA timezone name."""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}; use an IANA name such as Europe/Zagreb") from exc
        return value

    def to_domain(self) -> Schedule:
        """Return the domain schedule."""
        return Schedule(zone=ZoneInfo(self.timezone), windows=tuple(window.to_domain() for window in self.windows))


class DockerTargetSettings(_Section):
    """Docker compute target: a socket proxy that allows only container inspect, start and stop."""

    endpoint: str = "unix:///var/run/docker-proxy.sock"
    container: str | None = "voice-mcp-gpu-worker"
    request_timeout_seconds: float = Field(default=10, gt=0)


class AwsEcsTargetSettings(_Section):
    """Amazon ECS compute target: one service whose desired count is 0 or 1."""

    region: str | None = None
    cluster: str | None = None
    service: str | None = None


class OnDemandSettings(_Section):
    """Wake controller: starts the GPU worker for work or schedule windows and drains it when idle."""

    enabled: bool = False
    """Master switch; false constructs no controller, compute target, Docker or AWS client."""

    target: Literal["none", "docker", "aws_ecs"] = "none"
    dry_run: bool = False
    tick_seconds: float = Field(default=15, gt=0)
    idle_timeout_seconds: float = Field(default=300, ge=0)
    min_uptime_seconds: float = Field(default=300, ge=0)
    start_timeout_seconds: float = Field(default=900, gt=0)
    drain_confirm_margin_seconds: float = Field(default=60, ge=0)
    start_backoff_seconds: tuple[float, ...] = Field(default=(30, 60, 120, 300), min_length=1)
    schedule: ScheduleSettings = ScheduleSettings()
    docker: DockerTargetSettings = DockerTargetSettings()
    aws_ecs: AwsEcsTargetSettings = AwsEcsTargetSettings()

    @model_validator(mode="after")
    def _check_target(self) -> Self:
        """Require a target when enabled, and the settings of the selected target."""
        if self.enabled and self.target == "none":
            raise ValueError(
                "gpu_worker.on_demand.enabled requires target docker or aws_ecs; "
                "set dry_run: true to observe decisions without acting"
            )
        if self.target == "docker" and not self.docker.container:
            raise ValueError("gpu_worker.on_demand.target=docker requires docker.container")
        ecs = self.aws_ecs
        if self.target == "aws_ecs" and not (ecs.region and ecs.cluster and ecs.service):
            raise ValueError("gpu_worker.on_demand.target=aws_ecs requires aws_ecs.region, cluster and service")
        return self


class GpuWorkerSettings(_Section):
    """The GPU worker's idle unloading, draining and on-demand start."""

    idle_unload_seconds: float = Field(default=0, ge=0)
    """Unload models idle this long inside a running worker; 0 disables unloading (also applies to the all role)."""

    drain_timeout_seconds: float = Field(default=1800, gt=0)
    """How long a draining worker may finish its current job before it is released back to the queue."""

    on_demand: OnDemandSettings = OnDemandSettings()


class DatabaseSettings(_Section):
    """PostgreSQL connection settings; the URL comes only from the environment."""

    url: SecretStr | None = None
    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=5, ge=0)
    connect_timeout_seconds: float = Field(default=10, gt=0)
    statement_timeout_ms: int = Field(default=30000, ge=0)


class ModelsSettings(_Section):
    """Installed model location; runtime downloads stay disabled."""

    root: Path | None = None
    allow_runtime_downloads: Literal[False] = False


class ObservabilitySettings(_Section):
    """Logging settings."""

    log_format: Literal["json", "console"] = "json"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_job_events: bool = True


class SttPreset(_Section):
    """STT values a preset supplies; None leaves the default in place."""

    model: str | None = None
    beam_size: int | None = None
    temperature: float | None = None
    word_timestamps: bool | None = None
    segment_flags: bool | None = None
    denoise: bool | None = None
    condition_on_previous_text: bool | None = None
    vad_threshold: float | None = None

    def to_options(self) -> SttOptions:
        """Return the preset as domain STT options."""
        return SttOptions(
            beam_size=self.beam_size,
            temperature=self.temperature,
            word_timestamps=self.word_timestamps,
            denoise=self.denoise,
            condition_on_previous_text=self.condition_on_previous_text,
        )


class TtsPreset(_Section):
    """TTS values a preset supplies."""

    loudness_normalization: bool | None = None
    max_chunk_chars: int | None = None
    gap_ms: int | None = None

    def to_options(self) -> TtsOptions:
        """Return the preset as domain TTS options."""
        return TtsOptions(
            loudness_normalization=self.loudness_normalization, max_chunk_chars=self.max_chunk_chars, gap_ms=self.gap_ms
        )


class Preset(_Section):
    """Named quality preset from app/config/presets/."""

    description: str = ""
    stt: SttPreset = SttPreset()
    tts: TtsPreset = TtsPreset()


def _config_text(*parts: str) -> str:
    """Return the text of a packaged config file."""
    return resources.files("app.config").joinpath(*parts).read_text(encoding="utf-8")


def select_preset(presets: Mapping[str, Preset], name: str | None) -> Preset:
    """Return the named preset, an empty preset for None, or raise invalid_request for unknown names."""
    if name is None:
        return Preset()
    preset = presets.get(name)
    if preset is None:
        raise InvalidRequestError(f"unknown preset {name!r}; available: {', '.join(sorted(presets))}")
    return preset


def load_presets() -> dict[str, Preset]:
    """Return every packaged preset by name."""
    presets_dir = resources.files("app.config").joinpath("presets")
    presets: dict[str, Preset] = {}
    for entry in presets_dir.iterdir():
        if entry.name.endswith(".yaml"):
            presets[entry.name.removesuffix(".yaml")] = Preset.model_validate(yaml.safe_load(entry.read_text()) or {})
    return presets


def load_model_registry() -> dict[str, Any]:
    """Return the raw backend-aware model registry."""
    data = yaml.safe_load(_config_text("models.yaml"))
    return data if isinstance(data, dict) else {}


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set a value in nested dictionaries, creating levels as needed."""
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


class _YamlDefaultsSource(PydanticBaseSettingsSource):
    """Settings source that reads the packaged default.yaml."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Return nothing per field; values come from __call__."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return the default YAML document."""
        data = yaml.safe_load(_config_text("default.yaml"))
        return data if isinstance(data, dict) else {}


class _ShortAliasSource(PydanticBaseSettingsSource):
    """Settings source for short variables such as WHISPER_BACKEND and DATABASE_URL, from the environment or .env."""

    def __init__(self, settings_cls: type[BaseSettings], *, dotenv: dict[str, str | None]) -> None:
        """Store the parsed .env values used when a variable is not in the environment."""
        super().__init__(settings_cls)
        self._dotenv = dotenv

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Return nothing per field; values come from __call__."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return nested values for every short alias set in the environment, falling back to .env."""
        for retired, replacement in RETIRED_ENV_ALIASES.items():
            if os.environ.get(retired) or self._dotenv.get(retired):
                raise ValueError(f"{retired} was renamed to {replacement}; set {replacement} instead")
        values: dict[str, Any] = {}
        for variable, path in SHORT_ENV_ALIASES.items():
            raw = os.environ.get(variable) or self._dotenv.get(variable)
            if raw:
                _set_nested(values, path, raw)
        return values


class Settings(BaseSettings):
    """Application settings, validated once at startup and injected."""

    model_config = SettingsConfigDict(
        env_prefix="VOICE_MCP_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    server: ServerSettings = ServerSettings()
    auth: AuthConfig = AuthConfig()
    hardware: HardwareSettings = HardwareSettings()
    stt: SttSettings = SttSettings()
    tts: TtsSettings = TtsSettings()
    audio: AudioSettings = AudioSettings()
    jobs: JobsSettings = JobsSettings()
    storage: StorageSettings = StorageSettings()
    database: DatabaseSettings = DatabaseSettings()
    models: ModelsSettings = ModelsSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    deployment: DeploymentSettings = DeploymentSettings()
    gpu_worker: GpuWorkerSettings = GpuWorkerSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order sources: init, environment, short aliases, .env, default YAML."""
        env_file = settings_cls.model_config.get("env_file")
        dotenv = dotenv_values(env_file) if isinstance(env_file, str) and Path(env_file).is_file() else {}
        return (
            init_settings,
            env_settings,
            _ShortAliasSource(settings_cls, dotenv=dotenv),
            dotenv_settings,
            _YamlDefaultsSource(settings_cls),
        )

    @model_validator(mode="after")
    def _check_cross_fields(self) -> Self:
        """Validate combinations that depend on several sections."""
        if self.server.transport == "streamable_http":
            if self.auth.mode != "oauth":
                raise ValueError("streamable_http requires auth.mode=oauth; trusted_local is for stdio only")
            if not self.auth.issuer_url or not self.auth.audience:
                raise ValueError("streamable_http requires auth.issuer_url and auth.audience")
            if self.auth.audience != self.auth.resource_server_url:
                raise ValueError("auth.audience must equal auth.resource_server_url")
            if not self.auth.validate_token_resource:
                raise ValueError("auth.validate_token_resource must stay true")
        if self.storage.backend == "s3" and not (self.storage.s3.bucket and self.storage.s3.region):
            raise ValueError("storage.backend=s3 requires storage.s3.bucket and storage.s3.region")
        if self.gpu_worker.on_demand.enabled and self.deployment.role != "api":
            raise ValueError(
                f"gpu_worker.on_demand.enabled requires deployment.role=api (got {self.deployment.role}); "
                "a single all process cannot start or stop itself"
            )
        return self

    @property
    def data_root(self) -> Path:
        """Return the absolute runtime data root."""
        return self.storage.data_root.expanduser().resolve()

    @property
    def models_root(self) -> Path:
        """Return the absolute installed-models root."""
        return (self.models.root or self.data_root / "models").expanduser().resolve()

    @property
    def assets_root(self) -> Path:
        """Return the absolute local assets root."""
        return (self.storage.local.assets_root or self.data_root / "assets").expanduser().resolve()

    @property
    def workspace_root(self) -> Path:
        """Return the absolute workspace root."""
        return (self.storage.workspace_root or self.data_root / "workspace").expanduser().resolve()

    @property
    def input_directories(self) -> tuple[Path, ...]:
        """Return the absolute, resolved allowlisted input directories."""
        directories = self.server.allowed_input_directories or (self.data_root / "inputs",)
        return tuple(directory.expanduser().resolve() for directory in directories)


def load_settings(**overrides: Any) -> Settings:
    """Load and validate settings; keyword overrides take the highest precedence."""
    return Settings(**overrides)
