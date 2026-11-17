"""Pass-through denoiser."""

import shutil
from pathlib import Path

from app.domain.models import InstalledModel


class MockDenoiser:
    """Copies input to output and records calls."""

    def __init__(self) -> None:
        """Start unloaded."""
        self.loaded: InstalledModel | None = None
        self.calls: list[tuple[Path, Path]] = []

    @property
    def device(self) -> str:
        """Return the CPU device."""
        return "cpu"

    def load(self, model: InstalledModel) -> None:
        """Record the model."""
        self.loaded = model

    def denoise(self, source: Path, destination: Path) -> None:
        """Copy the file unchanged."""
        self.calls.append((source, destination))
        shutil.copyfile(source, destination)

    def unload(self) -> None:
        """Forget the model."""
        self.loaded = None
