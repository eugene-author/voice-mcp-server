"""Builds the model catalog from the packaged registry."""

from app.adapters.models.huggingface.catalog import HuggingFaceModelCatalog, parse_registry
from app.core.settings import Settings, load_model_registry


def build_model_catalog(settings: Settings) -> HuggingFaceModelCatalog:
    """Return the catalog rooted at the configured models root."""
    return HuggingFaceModelCatalog(root=settings.models_root, entries=parse_registry(load_model_registry()))
