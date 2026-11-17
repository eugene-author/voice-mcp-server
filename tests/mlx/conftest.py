"""Settings for real MLX tests on macOS runners; models are installed by the CI job."""

from pathlib import Path

from app.core.settings import Settings, load_settings

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "demo"


def mlx_settings(tmp_path: Path) -> Settings:
    """Return stdio settings using the repository's var/ models and a temporary data root."""
    return load_settings(
        server={"transport": "stdio"},
        storage={"data_root": str(tmp_path)},
        models={"root": str(ROOT / "var" / "models")},
    )
