"""Backend-aware model registry with local manifests; install is the only code path that downloads weights."""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.errors import InvalidRequestError, ModelNotInstalledError
from app.domain.models import InstalledModel, ModelEntry, ModelKind, TtsVariant

logger = logging.getLogger(__name__)
MANIFEST_NAME = "manifest.json"


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_registry(raw: Mapping[str, Any]) -> tuple[ModelEntry, ...]:
    """Return registry entries from the models.yaml document."""
    entries: list[ModelEntry] = []
    for kind_name, backends in raw.items():
        kind = ModelKind(kind_name)
        for backend, models in (backends or {}).items():
            for name, spec in (models or {}).items():
                entries.append(
                    ModelEntry(
                        name=name,
                        kind=kind,
                        backend=backend,
                        source=spec["source"],
                        approx_size_mb=int(spec.get("approx_size_mb", 0)),
                        min_memory_mb=int(spec.get("min_memory_mb", 0)),
                        variant=TtsVariant(spec["variant"]) if spec.get("variant") else None,
                        revision=spec.get("revision"),
                    )
                )
    return tuple(entries)


class HuggingFaceModelCatalog:
    """ModelCatalog over ${MODELS_ROOT}/<kind>/<backend>/<name>/ directories with manifest.json files."""

    def __init__(self, *, root: Path, entries: tuple[ModelEntry, ...]) -> None:
        """Store the models root and registry entries."""
        self._root = root
        self._entry_by_key = {(entry.kind, entry.backend, entry.name): entry for entry in entries}

    def entries(self, kind: ModelKind | None = None) -> tuple[ModelEntry, ...]:
        """Return registry entries, optionally of one kind."""
        return tuple(entry for entry in self._entry_by_key.values() if kind is None or entry.kind is kind)

    def installed(self, kind: ModelKind | None = None) -> tuple[InstalledModel, ...]:
        """Return the registry models whose manifests exist under the models root."""
        found = (self.find(entry.kind, entry.backend, entry.name) for entry in self.entries(kind))
        return tuple(model for model in found if model is not None)

    def find(self, kind: ModelKind, backend: str, name: str) -> InstalledModel | None:
        """Return an installed model or None."""
        entry = self._entry_by_key.get((kind, backend, name))
        directory = self._directory(kind, backend, name)
        manifest_path = directory / MANIFEST_NAME
        if entry is None or not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return InstalledModel(
            name=name,
            kind=kind,
            backend=backend,
            local_path=directory,
            revision=manifest.get("revision"),
            variant=entry.variant,
        )

    def require(self, kind: ModelKind, backend: str, name: str) -> InstalledModel:
        """Return an installed model or raise ModelNotInstalledError."""
        model = self.find(kind, backend, name)
        if model is None:
            raise ModelNotInstalledError(f"{kind} model {name!r} is not installed for {backend}")
        return model

    def install(self, kind: ModelKind, backend: str, name: str, *, source_dir: Path | None = None) -> InstalledModel:
        """Download (or copy from source_dir) a registry model into a staging directory, then move it into place."""
        entry = self._entry_by_key.get((kind, backend, name))
        if entry is None:
            raise InvalidRequestError(f"{kind} model {name!r} is not in the registry for {backend}")
        target = self._directory(kind, backend, name)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=target.parent))
        try:
            revision = self._populate(entry, staging, source_dir)
            self._write_manifest(
                entry, staging, revision=revision, origin=str(source_dir) if source_dir else entry.source
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        logger.info("model installed", extra={"kind": str(kind), "backend": backend, "model": name})
        return self.require(kind, backend, name)

    def _directory(self, kind: ModelKind, backend: str, name: str) -> Path:
        """Return the install directory of a model."""
        return self._root / str(kind) / backend / name

    def _populate(self, entry: ModelEntry, staging: Path, source_dir: Path | None) -> str | None:
        """Fill the staging directory and return the source revision when known."""
        if source_dir is not None:
            shutil.copytree(source_dir, staging, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache", ".git*"))
            return None
        return self._download_hf_snapshot(entry, staging)

    @staticmethod
    def _download_hf_snapshot(entry: ModelEntry, staging: Path) -> str | None:
        """Download a Hugging Face snapshot; huggingface_hub is imported here because only install may download."""
        from huggingface_hub import HfApi, snapshot_download

        snapshot_download(repo_id=entry.source, revision=entry.revision, local_dir=staging)
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        info = HfApi().model_info(entry.source, revision=entry.revision)
        return info.sha

    @staticmethod
    def _write_manifest(entry: ModelEntry, staging: Path, *, revision: str | None, origin: str) -> None:
        """Write manifest.json with file hashes, origin and install time."""
        files = {
            str(path.relative_to(staging)): _sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != MANIFEST_NAME
        }
        manifest = {
            "name": entry.name,
            "kind": str(entry.kind),
            "backend": entry.backend,
            "source": entry.source,
            "origin": origin,
            "revision": revision,
            "files": files,
            "installed_at": datetime.now(tz=UTC).isoformat(),
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
