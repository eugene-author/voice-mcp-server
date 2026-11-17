"""GPU worker registry values and the capability snapshot the API validates requests against."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from enum import StrEnum
from typing import Any

from app.domain.models import ModelKind
from app.domain.stt import SttBackendName, SttCapabilities, SttRuntime
from app.domain.tts import TtsBackendName, TtsCapabilities

PRELOAD_KINDS = frozenset(kind.value for kind in ModelKind)
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
"""Day names in datetime.weekday() order."""

LOOKAHEAD_DAYS = 8


class WorkerStatus(StrEnum):
    """Lifecycle of a GPU worker process as recorded in the registry."""

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A model named by kind and name, as in a preload entry such as stt:large-v3."""

    kind: ModelKind
    name: str

    @classmethod
    def parse(cls, raw: str) -> ModelRef:
        """Parse kind:name with kind in stt, tts or denoise; raise ValueError when malformed."""
        kind, separator, name = raw.partition(":")
        if not separator or kind not in PRELOAD_KINDS or not name.strip():
            raise ValueError(f"model reference {raw!r} must be kind:name with kind in {sorted(PRELOAD_KINDS)}")
        return cls(ModelKind(kind), name.strip())

    def __str__(self) -> str:
        """Return the kind:name form."""
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True, slots=True)
class InstalledModelRef:
    """An installed model identified by kind, backend and name."""

    kind: ModelKind
    backend: str
    name: str


@dataclass(frozen=True, slots=True)
class CapabilitySnapshot:
    """What a worker can run, published to the registry so the API validates requests without loading models."""

    profile: str
    stt_runtime: SttRuntime
    stt_capabilities: SttCapabilities
    tts_backend: TtsBackendName | None
    tts_device: str | None
    tts_capabilities: TtsCapabilities | None
    denoise_device: str | None
    """None when the worker has no denoiser."""

    installed: frozenset[InstalledModelRef]
    preset_speakers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Preset speakers per installed CustomVoice checkpoint name."""

    loaded: Mapping[str, str | None] = field(default_factory=dict)
    preload_errors: tuple[str, ...] = ()

    @property
    def denoise_available(self) -> bool:
        """Return whether the worker can denoise."""
        return self.denoise_device is not None

    def installed_names(self, kind: ModelKind, backend: str) -> frozenset[str]:
        """Return the installed model names of one kind and backend."""
        return frozenset(ref.name for ref in self.installed if ref.kind is kind and ref.backend == backend)

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot as JSON-serializable data."""
        tts = self.tts_capabilities
        return {
            "profile": self.profile,
            "stt_runtime": {
                "backend": str(self.stt_runtime.backend),
                "device": self.stt_runtime.device,
                "compute_type": self.stt_runtime.compute_type,
            },
            "stt_capabilities": _flags(self.stt_capabilities),
            "tts_backend": str(self.tts_backend) if self.tts_backend else None,
            "tts_device": self.tts_device,
            "tts_capabilities": _flags(tts) if tts else None,
            "denoise_device": self.denoise_device,
            "installed": sorted([str(ref.kind), ref.backend, ref.name] for ref in self.installed),
            "preset_speakers": {name: list(speakers) for name, speakers in self.preset_speakers.items()},
            "loaded": dict(self.loaded),
            "preload_errors": list(self.preload_errors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilitySnapshot:
        """Rebuild a snapshot from its to_dict form."""
        runtime = data["stt_runtime"]
        tts = data.get("tts_capabilities")
        return cls(
            profile=str(data["profile"]),
            stt_runtime=SttRuntime(
                backend=SttBackendName(runtime["backend"]),
                device=runtime["device"],
                compute_type=str(runtime["compute_type"]),
            ),
            stt_capabilities=SttCapabilities(**data["stt_capabilities"]),
            tts_backend=TtsBackendName(data["tts_backend"]) if data.get("tts_backend") else None,
            tts_device=data.get("tts_device"),
            tts_capabilities=TtsCapabilities(**tts) if tts else None,
            denoise_device=data.get("denoise_device"),
            installed=frozenset(
                InstalledModelRef(ModelKind(kind), str(backend), str(name)) for kind, backend, name in data["installed"]
            ),
            preset_speakers={str(name): tuple(speakers) for name, speakers in data.get("preset_speakers", {}).items()},
            loaded=dict(data.get("loaded", {})),
            preload_errors=tuple(data.get("preload_errors", ())),
        )


def _flags(capabilities: SttCapabilities | TtsCapabilities) -> dict[str, bool]:
    """Return a capability dataclass's boolean fields."""
    return {name: bool(value) for name, value in asdict(capabilities).items()}


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    """One row of the worker registry."""

    id: str
    """The worker's lease owner, host:pid."""

    status: WorkerStatus
    capabilities: CapabilitySnapshot
    started_at: datetime
    heartbeat_at: datetime
    last_activity_at: datetime | None = None
    drain_requested_at: datetime | None = None
    drained_at: datetime | None = None
    version: str = ""


@dataclass(frozen=True, slots=True)
class JobDemand:
    """Work the GPU worker has to do: jobs it could claim now and jobs it is executing."""

    runnable: int
    running: int


@dataclass(frozen=True, slots=True)
class ScheduleWindow:
    """A daily window, on some weekdays, during which the GPU worker stays warm; it starts lead early."""

    days: frozenset[int]
    """Weekdays as datetime.weekday() numbers, Monday = 0."""

    start: time
    end: time
    lead: timedelta = timedelta()
    preload: tuple[ModelRef, ...] = ()

    def span_on(self, day: date, zone: tzinfo) -> tuple[datetime, datetime] | None:
        """Return [start - lead, end) for a calendar day in zone, or None when the window skips that weekday."""
        if day.weekday() not in self.days:
            return None
        begin = datetime.combine(day, self.start, tzinfo=zone) - self.lead
        return begin, datetime.combine(day, self.end, tzinfo=zone)


@dataclass(frozen=True, slots=True)
class Schedule:
    """Schedule windows in one IANA timezone."""

    zone: tzinfo
    windows: tuple[ScheduleWindow, ...] = ()

    def active_window(self, now: datetime) -> ScheduleWindow | None:
        """Return the window whose [start - lead, end) contains now, if any."""
        local = now.astimezone(self.zone)
        for window in self.windows:
            for day in (local.date(), local.date() + timedelta(days=1)):
                span = window.span_on(day, self.zone)
                if span is not None and span[0] <= local < span[1]:
                    return window
        return None

    def next_start(self, now: datetime) -> datetime | None:
        """Return the earliest window start (lead included) after now, within the next week."""
        local = now.astimezone(self.zone)
        starts = [
            span[0]
            for window in self.windows
            for offset in range(LOOKAHEAD_DAYS)
            if (span := window.span_on(local.date() + timedelta(days=offset), self.zone)) is not None
            and span[0] > local
        ]
        return min(starts) if starts else None


class TargetState(StrEnum):
    """Observed state of the GPU worker's compute."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class WakeAction(StrEnum):
    """What the wake controller does on one tick."""

    NONE = "none"
    START = "start"
    REQUEST_DRAIN = "request_drain"
    CANCEL_DRAIN = "cancel_drain"
    STOP = "stop"
    FORCE_STOP = "force_stop"


@dataclass(frozen=True, slots=True)
class WakePolicy:
    """Timings of the wake controller, from gpu_worker settings."""

    idle_timeout: timedelta
    min_uptime: timedelta
    start_timeout: timedelta
    drain_timeout: timedelta
    drain_confirm_margin: timedelta
    lease_timeout: timedelta
    """A worker heartbeat older than this is not fresh (jobs.lease_timeout_seconds)."""


@dataclass(frozen=True, slots=True)
class ControllerObservation:
    """Everything one tick decides on: the database, the compute target, the schedule and the controller's memory."""

    now: datetime
    demand: JobDemand
    target: TargetState
    worker: WorkerRecord | None
    """The registry row with the newest heartbeat, fresh or not; None if no worker ever registered."""

    in_window: bool
    start_requested_at: datetime | None = None
    """When this controller last started the target and has not yet seen the worker ready."""

    target_running_since: datetime | None = None
    """When this controller first saw the target running in its current run."""

    backoff_until: datetime | None = None

    @property
    def new_demand(self) -> bool:
        """Return whether there is work the worker is not already executing: runnable jobs or a window."""
        return self.demand.runnable > 0 or self.in_window

    @property
    def any_demand(self) -> bool:
        """Return whether there is any demand: runnable or running jobs, or a window."""
        return self.new_demand or self.demand.running > 0


@dataclass(frozen=True, slots=True)
class Decision:
    """The action of one tick and why."""

    action: WakeAction
    reason: str
    start_failed: bool = False
    """True when the action force-stops a start that did not become ready, which advances the start backoff."""


READY_STATUSES = frozenset({WorkerStatus.READY, WorkerStatus.BUSY})


def backoff_delay(failures: int, steps: tuple[float, ...]) -> timedelta:
    """Return the wait after the given number of consecutive failed starts; the last step repeats."""
    if failures <= 0 or not steps:
        return timedelta()
    return timedelta(seconds=steps[min(failures, len(steps)) - 1])


def worker_fresh(observation: ControllerObservation, policy: WakePolicy) -> bool:
    """Return whether the newest worker row's heartbeat is within the lease timeout."""
    worker = observation.worker
    return worker is not None and observation.now - worker.heartbeat_at < policy.lease_timeout


def worker_ready(observation: ControllerObservation, policy: WakePolicy) -> bool:
    """Return whether a worker is up: fresh, and ready or busy."""
    worker = observation.worker
    return worker is not None and worker.status in READY_STATUSES and worker_fresh(observation, policy)


def decide(observation: ControllerObservation, policy: WakePolicy) -> Decision:
    """Return the wake controller's action for one observation; pure, so every rule is unit tested."""
    obs = observation
    if obs.target is TargetState.UNKNOWN:
        return Decision(WakeAction.NONE, "compute target state is unknown")
    started = obs.start_requested_at
    if (
        started is not None
        and obs.target in (TargetState.STARTING, TargetState.RUNNING)
        and not worker_ready(obs, policy)
        and obs.now - started >= policy.start_timeout
    ):
        return Decision(WakeAction.FORCE_STOP, "worker not ready within start_timeout_seconds", start_failed=True)
    if obs.target in (TargetState.STARTING, TargetState.STOPPING):
        return Decision(WakeAction.NONE, f"compute target is {obs.target}")
    if obs.target is TargetState.STOPPED:
        return _decide_stopped(obs)
    return _decide_running(obs, policy)


def _decide_stopped(obs: ControllerObservation) -> Decision:
    """Start for runnable jobs, a schedule window, or a first registration, unless a start backoff is pending."""
    if obs.demand.runnable > 0:
        reason = "runnable jobs are waiting"
    elif obs.in_window:
        reason = "a schedule window is active or within its lead time"
    elif obs.worker is None:
        reason = "no worker has registered yet"
    else:
        return Decision(WakeAction.NONE, "compute is stopped and there is no demand")
    if obs.backoff_until is not None and obs.now < obs.backoff_until:
        return Decision(WakeAction.NONE, f"{reason}, but a start backoff is pending")
    return Decision(WakeAction.START, reason)


def _decide_running(obs: ControllerObservation, policy: WakePolicy) -> Decision:
    """Decide for a running target from the worker's registry row."""
    worker = obs.worker
    if worker is None or not worker_fresh(obs, policy):
        return _decide_silent(obs, policy)
    if obs.start_requested_at is not None and worker.status in (WorkerStatus.STOPPING, WorkerStatus.STOPPED):
        return Decision(WakeAction.NONE, "waiting for the started worker to register")
    match worker.status:
        case WorkerStatus.DRAINING:
            return _decide_draining(obs, policy, worker)
        case WorkerStatus.STOPPING:
            return Decision(WakeAction.NONE, "worker is stopping")
        case WorkerStatus.STOPPED if obs.new_demand:
            return Decision(WakeAction.NONE, "worker stopped and new demand appeared; waiting for the target to stop")
        case WorkerStatus.STOPPED:
            return Decision(WakeAction.STOP, "worker drained and stopped, and there is no demand")
        case WorkerStatus.STARTING:
            return Decision(WakeAction.NONE, "worker is starting")
        case WorkerStatus.BUSY:
            return Decision(WakeAction.NONE, "a job is executing")
    if obs.demand.running > 0:
        return Decision(WakeAction.NONE, "a job is executing")
    if worker.drain_requested_at is not None:
        if obs.new_demand:
            return Decision(WakeAction.CANCEL_DRAIN, "new demand appeared before the worker began draining")
        return Decision(WakeAction.NONE, "drain requested; waiting for the worker")
    return _decide_ready(obs, policy, worker)


def _decide_silent(obs: ControllerObservation, policy: WakePolicy) -> Decision:
    """Force-stop a running target whose worker has not heartbeated for longer than the start timeout."""
    if obs.start_requested_at is not None:
        return Decision(WakeAction.NONE, "waiting for the started worker to register")
    seen = [moment for moment in (obs.target_running_since, obs.worker and obs.worker.heartbeat_at) if moment]
    if seen and obs.now - max(seen) > policy.start_timeout:
        return Decision(WakeAction.FORCE_STOP, "no fresh worker heartbeat for longer than start_timeout_seconds")
    return Decision(WakeAction.NONE, "waiting for a fresh worker heartbeat")


def _decide_draining(obs: ControllerObservation, policy: WakePolicy, worker: WorkerRecord) -> Decision:
    """Force-stop an overdue drain, cancel a controller drain on new demand, or wait."""
    requested = worker.drain_requested_at
    if requested is not None and obs.now - requested > policy.drain_timeout + policy.drain_confirm_margin:
        return Decision(WakeAction.FORCE_STOP, "draining longer than drain_timeout_seconds plus the margin")
    if requested is not None and obs.new_demand:
        return Decision(WakeAction.CANCEL_DRAIN, "new demand appeared while draining")
    return Decision(WakeAction.NONE, "worker is draining")


def _decide_ready(obs: ControllerObservation, policy: WakePolicy, worker: WorkerRecord) -> Decision:
    """Request a drain once a ready worker has been idle long enough, no sooner than min_uptime after a start."""
    if obs.in_window:
        return Decision(WakeAction.NONE, "staying warm for the schedule window")
    if obs.demand.runnable > 0:
        return Decision(WakeAction.NONE, "runnable jobs are waiting for the worker")
    if obs.now - worker.started_at < policy.min_uptime:
        return Decision(WakeAction.NONE, "within min_uptime_seconds of the start")
    last_activity = max(moment for moment in (worker.last_activity_at, worker.started_at) if moment)
    if obs.now - last_activity < policy.idle_timeout:
        return Decision(WakeAction.NONE, "worker was active recently")
    return Decision(WakeAction.REQUEST_DRAIN, "worker idle for idle_timeout_seconds")
