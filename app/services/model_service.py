"""Model registry views for list_models: installation, active backend and loaded state per entry."""

from dataclasses import dataclass

from app.domain.errors import CapabilityUnavailableError
from app.domain.models import ModelKind
from app.ports.models import ModelCatalog
from app.ports.runtime import RuntimeCatalog


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Registry entry with installation and activity state."""

    name: str
    kind: ModelKind
    backend: str
    installed: bool
    active_backend: bool
    """Whether the entry targets the backend or runtime active on this host."""

    loaded: bool
    approx_size_mb: int
    source: str
    variant: str | None


class ModelService:
    """Answers model queries; installing is the CLI's job through the model catalog."""

    def __init__(self, *, catalog: ModelCatalog, runtime: RuntimeCatalog) -> None:
        """Store the registry and the runtime catalog that knows what is installed, active and resident."""
        self._catalog = catalog
        self._runtime = runtime

    def list_models(self, kind: ModelKind | None = None) -> list[ModelStatus]:
        """Return registry entries with installed and active flags; installed active entries first."""
        active, loaded = self._active_backends()
        statuses = [
            ModelStatus(
                name=entry.name,
                kind=entry.kind,
                backend=entry.backend,
                installed=bool(active) and self._runtime.has_model(entry.kind, entry.backend, entry.name),
                active_backend=active.get(entry.kind) == entry.backend,
                loaded=loaded.get(str(entry.kind)) == entry.name and active.get(entry.kind) == entry.backend,
                approx_size_mb=entry.approx_size_mb,
                source=entry.source,
                variant=str(entry.variant) if entry.variant else None,
            )
            for entry in self._catalog.entries(kind)
        ]
        return sorted(statuses, key=lambda s: (not (s.installed and s.active_backend), str(s.kind), s.backend, s.name))

    def _active_backends(self) -> tuple[dict[ModelKind, str | None], dict[str, str | None]]:
        """Return the active backend per kind and the resident models; both empty while no runtime is known."""
        try:
            tts = self._runtime.tts_backend()
            active: dict[ModelKind, str | None] = {
                ModelKind.STT: str(self._runtime.stt_runtime().backend),
                ModelKind.TTS: str(tts) if tts else None,
                ModelKind.DENOISE: "deepfilternet" if self._runtime.denoise_available() else None,
            }
            return active, self._runtime.loaded_models()
        except CapabilityUnavailableError:
            return {}, {}
