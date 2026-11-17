"""In-memory model catalog."""

from pathlib import Path

from app.domain.errors import InvalidRequestError
from app.domain.models import InstalledModel, ModelEntry, ModelKind, TtsVariant

DEFAULT_ENTRIES = (
    ModelEntry("tiny", ModelKind.STT, "ctranslate2", "Systran/faster-whisper-tiny", 75, 512),
    ModelEntry("small", ModelKind.STT, "ctranslate2", "Systran/faster-whisper-small", 484, 2048),
    ModelEntry("medium", ModelKind.STT, "ctranslate2", "Systran/faster-whisper-medium", 1530, 5120),
    ModelEntry("small", ModelKind.STT, "mlx", "mlx-community/whisper-small-mlx", 484, 2048),
    ModelEntry(
        "Qwen3-TTS-12Hz-0.6B-Base", ModelKind.TTS, "torch", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", 2500, 4096, TtsVariant.BASE
    ),
    ModelEntry(
        "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ModelKind.TTS,
        "torch",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        2500,
        4096,
        TtsVariant.CUSTOM_VOICE,
    ),
    ModelEntry("deepfilternet3", ModelKind.DENOISE, "deepfilternet", "github:x", 8, 256),
)


class MockModelCatalog:
    """Catalog whose installed set is configured directly."""

    def __init__(self, *, installed: set[tuple[ModelKind, str, str]], root: Path = Path("/models")) -> None:
        """Configure installed (kind, backend, name) triples."""
        self._installed = set(installed)
        self._root = root
        self._entries = DEFAULT_ENTRIES

    def entries(self, kind: ModelKind | None = None) -> tuple[ModelEntry, ...]:
        """Return registry entries."""
        return tuple(e for e in self._entries if kind is None or e.kind is kind)

    def installed(self, kind: ModelKind | None = None) -> tuple[InstalledModel, ...]:
        """Return installed models."""
        found = (self.find(e.kind, e.backend, e.name) for e in self.entries(kind))
        return tuple(model for model in found if model is not None)

    def find(self, kind: ModelKind, backend: str, name: str) -> InstalledModel | None:
        """Return an installed model or None."""
        entry = next((e for e in self._entries if (e.kind, e.backend, e.name) == (kind, backend, name)), None)
        if entry is None or (kind, backend, name) not in self._installed:
            return None
        return InstalledModel(name, kind, backend, self._root / str(kind) / backend / name, None, entry.variant)

    def install(self, kind: ModelKind, backend: str, name: str, *, source_dir: Path | None = None) -> InstalledModel:
        """Mark a registry model installed."""
        if not any((e.kind, e.backend, e.name) == (kind, backend, name) for e in self._entries):
            raise InvalidRequestError("unknown model")
        self._installed.add((kind, backend, name))
        installed = self.find(kind, backend, name)
        assert installed is not None
        return installed
