"""Pure runtime policy for backends, devices and options; unsupported caller options fail, preset ones adapt."""

from collections.abc import Collection
from dataclasses import replace
from typing import Literal

from app.domain.errors import BackendUnavailableError, InvalidRequestError, ModelNotInstalledError
from app.domain.host import HostProfile
from app.domain.stt import (
    ResolvedSttOptions,
    SttBackendChoice,
    SttBackendName,
    SttCapabilities,
    SttOptions,
    SttRuntime,
    Substitution,
)
from app.domain.tts import OUTPUT_FORMATS, ResolvedTtsOptions, TtsBackendChoice, TtsBackendName, TtsOptions

LOW_VRAM_GPU_MB = 8192


def select_stt_backend(choice: SttBackendChoice, host: HostProfile) -> SttBackendName:
    """Resolve the configured STT backend choice for this host."""
    supports_mlx = host.is_apple_silicon and host.mlx_available
    match choice:
        case SttBackendChoice.MLX:
            if not supports_mlx:
                raise BackendUnavailableError("mlx requires macOS on Apple Silicon with the 'mlx' extra installed")
            return SttBackendName.MLX
        case SttBackendChoice.CTRANSLATE2:
            return SttBackendName.CTRANSLATE2
        case SttBackendChoice.AUTO:
            return SttBackendName.MLX if supports_mlx else SttBackendName.CTRANSLATE2


def select_ct2_runtime(
    host: HostProfile, *, device: Literal["auto", "cuda", "cpu"], compute_type: str, low_vram_threshold_mb: int
) -> SttRuntime:
    """Return the CTranslate2 device and compute type, failing on unsupported explicit choices."""
    has_cuda = host.cuda_device_count > 0 and host.system != "Darwin"
    if device == "cuda" and not has_cuda:
        raise BackendUnavailableError("stt.device=cuda but CTranslate2 reports no CUDA device")
    use_cuda = device == "cuda" or (device == "auto" and has_cuda and "float16" in host.cuda_compute_types)
    if not use_cuda:
        resolved = "int8" if compute_type == "auto" else compute_type
        if resolved not in {"int8", "int8_float32", "float32", "int16"}:
            raise BackendUnavailableError(f"compute type {resolved} is not supported on CPU")
        return SttRuntime(backend=SttBackendName.CTRANSLATE2, device="cpu", compute_type=resolved)
    if compute_type == "auto":
        low_vram = (host.cuda_memory_mb or 0) < low_vram_threshold_mb
        preferred = "int8_float16" if low_vram else "float16"
        resolved = preferred if preferred in host.cuda_compute_types else "float32"
    else:
        resolved = compute_type
    if host.cuda_compute_types and resolved not in host.cuda_compute_types:
        raise BackendUnavailableError(f"compute type {resolved} is not supported by the CUDA device")
    return SttRuntime(backend=SttBackendName.CTRANSLATE2, device="cuda", compute_type=resolved)


def select_tts_backend(choice: TtsBackendChoice, host: HostProfile) -> TtsBackendName | None:
    """Resolve the TTS runtime; None means no runtime is installed and TTS is unavailable."""
    supports_mlx = host.is_apple_silicon and host.mlx_audio_available
    match choice:
        case TtsBackendChoice.MLX:
            if not supports_mlx:
                raise BackendUnavailableError("mlx TTS requires macOS on Apple Silicon with the 'mlx' extra installed")
            return TtsBackendName.MLX
        case TtsBackendChoice.TORCH:
            if not host.qwen_tts_available:
                raise BackendUnavailableError("TTS_BACKEND=torch requires the 'tts' extra (qwen-tts)")
            return TtsBackendName.TORCH
        case TtsBackendChoice.AUTO:
            if supports_mlx:
                return TtsBackendName.MLX
            return TtsBackendName.TORCH if host.qwen_tts_available else None


def select_torch_device(host: HostProfile, *, allow_cpu: bool) -> Literal["cuda", "cpu"]:
    """Return the PyTorch device for TTS or denoising."""
    if host.torch_cuda_available:
        return "cuda"
    if not allow_cpu:
        raise BackendUnavailableError("no PyTorch CUDA device and tts.allow_cpu is false")
    return "cpu"


def platform_profile(host: HostProfile, *, devices: tuple[str | None, ...]) -> str:
    """Return macos-native, linux-cuda when any component actually runs on CUDA, otherwise linux-cpu."""
    if host.system == "Darwin":
        return "macos-native"
    return "linux-cuda" if "cuda" in devices else "linux-cpu"


def exclusive_residency(host: HostProfile, *, ram_threshold_mb: int) -> bool:
    """Return whether only one model may stay loaded at a time on this host."""
    if host.total_ram_mb <= ram_threshold_mb:
        return True
    gpu_memory = host.cuda_memory_mb
    return gpu_memory is not None and host.torch_cuda_available and gpu_memory < LOW_VRAM_GPU_MB


def resolve_stt_model(
    *, requested: str | None, preset_model: str | None, default: str, installed: Collection[str]
) -> tuple[str, tuple[Substitution, ...]]:
    """Return the model to use; an uninstalled caller model fails while an uninstalled preset model is substituted."""
    if requested is not None:
        if requested not in installed:
            raise ModelNotInstalledError(f"STT model {requested!r} is not installed for the active backend")
        return requested, ()
    if preset_model is not None and preset_model != default:
        if preset_model in installed:
            return preset_model, ()
        if default in installed:
            return default, (Substitution("model", preset_model, default, "is not installed"),)
    if default not in installed:
        raise ModelNotInstalledError(f"STT model {default!r} is not installed for the active backend")
    return default, ()


def resolve_stt_options(
    *,
    requested: SttOptions | None,
    preset: SttOptions | None,
    defaults: ResolvedSttOptions,
    capabilities: SttCapabilities,
    backend: SttBackendName,
) -> tuple[ResolvedSttOptions, tuple[Substitution, ...]]:
    """Merge caller, preset and default options; reject unsupported caller options, substitute the rest."""
    caller = requested or SttOptions()
    _reject_unsupported(caller, capabilities, backend)
    layered = caller.merged_over(preset or SttOptions())
    resolved = _apply(defaults, layered)
    return _substitute_unsupported(resolved, capabilities, backend)


def _reject_unsupported(caller: SttOptions, capabilities: SttCapabilities, backend: SttBackendName) -> None:
    """Raise InvalidRequestError for caller-supplied options the backend cannot honor."""
    problems: list[str] = []
    if caller.beam_size is not None and caller.beam_size > 1 and not capabilities.beam_search:
        problems.append("beam_size > 1")
    if caller.hotwords and not capabilities.hotwords:
        problems.append("hotwords")
    if caller.initial_prompt and not capabilities.initial_prompt:
        problems.append("initial_prompt")
    if caller.word_timestamps and not capabilities.word_timestamps:
        problems.append("word_timestamps")
    if problems:
        raise InvalidRequestError(f"{', '.join(problems)} not supported by the {backend} backend")


def _apply(defaults: ResolvedSttOptions, layered: SttOptions) -> ResolvedSttOptions:
    """Return defaults overridden by every set value of layered."""
    overrides = {name: getattr(layered, name) for name in layered.specified() if name != "vad"}
    resolved = replace(defaults, **overrides)
    if layered.vad is not None:
        resolved = replace(resolved, vad=replace(resolved.vad, enabled=layered.vad))
    return resolved


def _substitute_unsupported(
    resolved: ResolvedSttOptions, capabilities: SttCapabilities, backend: SttBackendName
) -> tuple[ResolvedSttOptions, tuple[Substitution, ...]]:
    """Replace preset or default values the backend cannot honor and record each substitution."""
    substitutions: list[Substitution] = []
    reason = f"not supported by {backend} backend"
    if resolved.beam_size > 1 and not capabilities.beam_search:
        substitutions.append(Substitution("beam_size", resolved.beam_size, 1, f"{reason} (greedy decoding)"))
        resolved = replace(resolved, beam_size=1)
    if resolved.hotwords and not capabilities.hotwords:
        substitutions.append(Substitution("hotwords", list(resolved.hotwords), [], reason))
        resolved = replace(resolved, hotwords=())
    if resolved.initial_prompt and not capabilities.initial_prompt:
        substitutions.append(Substitution("initial_prompt", resolved.initial_prompt, None, reason))
        resolved = replace(resolved, initial_prompt=None)
    if resolved.word_timestamps and not capabilities.word_timestamps:
        substitutions.append(Substitution("word_timestamps", True, False, reason))
        resolved = replace(resolved, word_timestamps=False)
    return resolved, tuple(substitutions)


def resolve_tts_options(
    *, requested: TtsOptions | None, preset: TtsOptions | None, defaults: ResolvedTtsOptions
) -> ResolvedTtsOptions:
    """Merge caller, preset and default synthesis options and validate the output format."""
    resolved = defaults
    for layer in (preset, requested):
        if layer is not None:
            resolved = replace(resolved, **{name: getattr(layer, name) for name in layer.specified()})
    if resolved.output_format not in OUTPUT_FORMATS:
        raise InvalidRequestError(f"output_format must be one of {', '.join(OUTPUT_FORMATS)}")
    if not 50 <= resolved.max_chunk_chars <= 2000:
        raise InvalidRequestError("max_chunk_chars must be between 50 and 2000")
    if not 0 <= resolved.gap_ms <= 2000:
        raise InvalidRequestError("gap_ms must be between 0 and 2000")
    return resolved
