"""Model registry and installed-model catalog port."""

from pathlib import Path
from typing import Protocol

from app.domain.models import InstalledModel, ModelEntry, ModelKind


class ModelCatalog(Protocol):
    """Backend-aware model registry and local installation state."""

    def entries(self, kind: ModelKind | None = None) -> tuple[ModelEntry, ...]:
        """Return registry entries, optionally of one kind."""
        ...

    def installed(self, kind: ModelKind | None = None) -> tuple[InstalledModel, ...]:
        """Return the models installed under the models root."""
        ...

    def find(self, kind: ModelKind, backend: str, name: str) -> InstalledModel | None:
        """Return an installed model or None."""
        ...

    def install(self, kind: ModelKind, backend: str, name: str, *, source_dir: Path | None = None) -> InstalledModel:
        """Download (or copy from source_dir) a registry model and write its manifest."""
        ...
