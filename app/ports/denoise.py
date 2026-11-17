"""Neural noise-reduction port."""

from pathlib import Path
from typing import Protocol

from app.domain.models import InstalledModel


class Denoiser(Protocol):
    """Blocking denoiser, always called on the inference executor thread."""

    @property
    def device(self) -> str:
        """Return the device the denoiser runs on."""
        ...

    def load(self, model: InstalledModel) -> None:
        """Load the installed denoise model from its local directory."""
        ...

    def denoise(self, source: Path, destination: Path) -> None:
        """Write a denoised copy of a mono WAV file, keeping its sample rate."""
        ...

    def unload(self) -> None:
        """Release the loaded model."""
        ...
