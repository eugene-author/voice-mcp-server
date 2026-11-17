"""Resolves the Whisper model and options of a transcription from the caller, the preset and server defaults."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from app.core.settings import Preset, Settings, select_preset
from app.domain.errors import InvalidRequestError
from app.domain.jobs import TranscriptionSpec
from app.domain.speech_text import normalize_language
from app.domain.stt import STT_MODEL_NAMES, ResolvedSttOptions, SttOptions, Substitution
from app.ports.runtime import RuntimeCatalog
from app.services.runtime_selection import resolve_stt_model, resolve_stt_options


@dataclass(frozen=True, slots=True)
class ResolvedTranscription:
    """A frozen transcription spec and every substitution made while resolving it (reported as job warnings)."""

    spec: TranscriptionSpec
    substitutions: tuple[Substitution, ...]


class TranscriptionOptionsResolver:
    """Applies caller > preset > default precedence for single files and batch children against installed models."""

    def __init__(
        self,
        *,
        settings: Settings,
        presets: Mapping[str, Preset],
        runtime: RuntimeCatalog,
    ) -> None:
        """Store settings, presets and the runtime catalog that answers model and capability questions."""
        self._settings = settings
        self._presets = presets
        self._runtime = runtime

    def resolve(
        self,
        *,
        input_path: Path,
        model: str | None,
        language: str | None,
        preset_name: str | None,
        options: SttOptions | None,
    ) -> ResolvedTranscription:
        """Return the spec for one input file; raise invalid_request for unknown models, presets or languages."""
        if model is not None and model not in STT_MODEL_NAMES:
            raise InvalidRequestError(f"model must be one of {', '.join(STT_MODEL_NAMES)}")
        preset = select_preset(self._presets, preset_name)
        runtime = self._runtime.stt_runtime()
        resolved_model, model_substitutions = resolve_stt_model(
            requested=model,
            preset_model=preset.stt.model,
            default=self._settings.stt.model,
            installed=self._runtime.installed_stt_names(),
        )
        resolved, option_substitutions = resolve_stt_options(
            requested=options,
            preset=preset.stt.to_options(),
            defaults=self._defaults(preset),
            capabilities=self._runtime.stt_capabilities(),
            backend=runtime.backend,
        )
        denoise_substitutions: tuple[Substitution, ...] = ()
        if resolved.denoise and not self._runtime.denoise_available():
            denoise_substitutions = (Substitution("denoise", True, False, "skipped: DeepFilterNet is not installed"),)
            resolved = replace(resolved, denoise=False)
        spec = TranscriptionSpec(
            input_path=str(input_path),
            model=resolved_model,
            language=normalize_language(language if language is not None else self._settings.stt.language),
            options=resolved,
            preserve_input=self._settings.storage.preserve_inputs,
        )
        return ResolvedTranscription(spec, (*model_substitutions, *option_substitutions, *denoise_substitutions))

    def _defaults(self, preset: Preset) -> ResolvedSttOptions:
        """Return server default options with the preset values that have no caller equivalent applied."""
        stt, audio = self._settings.stt, self._settings.audio
        defaults = ResolvedSttOptions(
            beam_size=stt.beam_size,
            temperature=stt.temperature,
            word_timestamps=stt.word_timestamps,
            initial_prompt=", ".join(stt.glossary) or None,
            vad=stt.vad.to_domain(),
            denoise=audio.denoising.enabled,
            remove_fillers=audio.filler_removal == "remove",
            remove_repetitions=audio.repeated_starts == "remove",
            segment_flags=stt.segment_flags,
        )
        if preset.stt.segment_flags is not None:
            defaults = replace(defaults, segment_flags=preset.stt.segment_flags)
        if preset.stt.vad_threshold is not None:
            defaults = replace(defaults, vad=replace(defaults.vad, threshold=preset.stt.vad_threshold))
        return defaults
