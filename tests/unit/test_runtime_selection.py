"""Pure backend, device and option selection for every host branch."""

import pytest

from app.domain.errors import BackendUnavailableError, InvalidRequestError, ModelNotInstalledError
from app.domain.host import HostProfile
from app.domain.stt import ResolvedSttOptions, SttBackendChoice, SttBackendName, SttOptions
from app.domain.tts import ResolvedTtsOptions, TtsBackendChoice, TtsBackendName, TtsOptions
from app.services.runtime_selection import (
    exclusive_residency,
    platform_profile,
    resolve_stt_model,
    resolve_stt_options,
    resolve_tts_options,
    select_ct2_runtime,
    select_stt_backend,
    select_torch_device,
    select_tts_backend,
)
from tests.mocks.host import cuda_host, host_profile, mac_host
from tests.mocks.stt import FULL_CAPABILITIES, MLX_LIKE_CAPABILITIES


@pytest.mark.parametrize(
    ("choice", "host", "expected"),
    [
        (SttBackendChoice.AUTO, host_profile(), SttBackendName.CTRANSLATE2),
        (SttBackendChoice.AUTO, cuda_host(), SttBackendName.CTRANSLATE2),
        (SttBackendChoice.AUTO, mac_host(), SttBackendName.MLX),
        (SttBackendChoice.AUTO, mac_host(mlx=False), SttBackendName.CTRANSLATE2),
        (SttBackendChoice.CTRANSLATE2, mac_host(), SttBackendName.CTRANSLATE2),
        (SttBackendChoice.MLX, mac_host(), SttBackendName.MLX),
    ],
)
def test_select_stt_backend(choice: SttBackendChoice, host: HostProfile, expected: SttBackendName) -> None:
    """Resolve the STT backend for each host."""
    assert select_stt_backend(choice, host) is expected


@pytest.mark.parametrize("host", [host_profile(), cuda_host(), mac_host(mlx=False)])
def test_forced_mlx_on_unsupported_host_fails(host: HostProfile) -> None:
    """Fail startup when mlx is forced where it cannot run."""
    with pytest.raises(BackendUnavailableError):
        select_stt_backend(SttBackendChoice.MLX, host)


def test_ct2_runtime_cpu_without_cuda() -> None:
    """Use CPU int8 without CUDA devices."""
    runtime = select_ct2_runtime(host_profile(), device="auto", compute_type="auto", low_vram_threshold_mb=4000)
    assert (runtime.device, runtime.compute_type) == ("cpu", "int8")


def test_ct2_runtime_cuda_float16() -> None:
    """Use CUDA float16 on a large GPU."""
    runtime = select_ct2_runtime(cuda_host(), device="auto", compute_type="auto", low_vram_threshold_mb=4000)
    assert (runtime.device, runtime.compute_type) == ("cuda", "float16")


def test_ct2_runtime_low_vram_uses_int8_float16() -> None:
    """Use int8_float16 below the VRAM threshold."""
    runtime = select_ct2_runtime(
        cuda_host(memory_mb=3072), device="auto", compute_type="auto", low_vram_threshold_mb=4000
    )
    assert runtime.compute_type == "int8_float16"


def test_ct2_runtime_gpu_without_float16_falls_back_to_cpu() -> None:
    """Prefer CPU when the GPU lacks float16 support, as on Maxwell cards."""
    host = cuda_host(compute_types=frozenset({"float32"}))
    runtime = select_ct2_runtime(host, device="auto", compute_type="auto", low_vram_threshold_mb=4000)
    assert runtime.device == "cpu"


def test_ct2_explicit_cuda_without_device_fails() -> None:
    """Fail on an explicit CUDA choice without CUDA devices."""
    with pytest.raises(BackendUnavailableError):
        select_ct2_runtime(host_profile(), device="cuda", compute_type="auto", low_vram_threshold_mb=4000)


def test_ct2_unsupported_explicit_compute_type_fails() -> None:
    """Fail on a compute type the GPU does not support."""
    host = cuda_host(compute_types=frozenset({"float32"}))
    with pytest.raises(BackendUnavailableError):
        select_ct2_runtime(host, device="cuda", compute_type="float16", low_vram_threshold_mb=4000)


def test_ct2_never_uses_cuda_on_macos() -> None:
    """Keep CTranslate2 on CPU on macOS."""
    host = mac_host()
    runtime = select_ct2_runtime(host, device="auto", compute_type="auto", low_vram_threshold_mb=4000)
    assert runtime.device == "cpu"


@pytest.mark.parametrize(
    ("choice", "host", "expected"),
    [
        (TtsBackendChoice.AUTO, host_profile(), TtsBackendName.TORCH),
        (TtsBackendChoice.AUTO, mac_host(), TtsBackendName.MLX),
        (TtsBackendChoice.AUTO, mac_host(mlx_audio=False, qwen_tts=True), TtsBackendName.TORCH),
        (TtsBackendChoice.AUTO, host_profile(qwen_tts_available=False), None),
        (TtsBackendChoice.TORCH, mac_host(qwen_tts=True), TtsBackendName.TORCH),
        (TtsBackendChoice.MLX, mac_host(), TtsBackendName.MLX),
    ],
)
def test_select_tts_backend(choice: TtsBackendChoice, host: HostProfile, expected: TtsBackendName | None) -> None:
    """Resolve the TTS runtime for each host."""
    assert select_tts_backend(choice, host) is expected


@pytest.mark.parametrize(
    ("choice", "host"),
    [
        (TtsBackendChoice.MLX, host_profile()),
        (TtsBackendChoice.MLX, mac_host(mlx_audio=False)),
        (TtsBackendChoice.TORCH, mac_host(qwen_tts=False)),
    ],
)
def test_forced_unsupported_tts_backend_fails(choice: TtsBackendChoice, host: HostProfile) -> None:
    """Fail startup for forced runtimes the host cannot run."""
    with pytest.raises(BackendUnavailableError):
        select_tts_backend(choice, host)


def test_torch_device_selection() -> None:
    """Use CUDA when available, CPU when allowed, otherwise fail."""
    assert select_torch_device(cuda_host(), allow_cpu=False) == "cuda"
    assert select_torch_device(host_profile(), allow_cpu=True) == "cpu"
    with pytest.raises(BackendUnavailableError):
        select_torch_device(host_profile(), allow_cpu=False)


def test_exclusive_residency_on_small_hosts() -> None:
    """Keep one model resident on 16 GB hosts and small GPUs."""
    assert exclusive_residency(host_profile(total_ram_mb=16384), ram_threshold_mb=16384)
    assert exclusive_residency(cuda_host(memory_mb=6000), ram_threshold_mb=16384)
    assert not exclusive_residency(cuda_host(memory_mb=24576), ram_threshold_mb=16384)


def test_caller_unsupported_options_are_rejected() -> None:
    """Reject caller-supplied options the backend cannot honor, naming the backend."""
    with pytest.raises(InvalidRequestError, match="beam_size > 1, hotwords not supported by the mlx backend"):
        resolve_stt_options(
            requested=SttOptions(beam_size=5, hotwords=("x",)),
            preset=None,
            defaults=ResolvedSttOptions(),
            capabilities=MLX_LIKE_CAPABILITIES,
            backend=SttBackendName.MLX,
        )


def test_preset_unsupported_values_are_substituted() -> None:
    """Substitute preset beam search with greedy decoding and record it."""
    resolved, substitutions = resolve_stt_options(
        requested=None,
        preset=SttOptions(beam_size=5),
        defaults=ResolvedSttOptions(beam_size=5),
        capabilities=MLX_LIKE_CAPABILITIES,
        backend=SttBackendName.MLX,
    )
    assert resolved.beam_size == 1
    assert [s.option for s in substitutions] == ["beam_size"]
    assert "greedy" in substitutions[0].describe()


def test_caller_overrides_preset_and_defaults() -> None:
    """Layer caller over preset over defaults."""
    resolved, substitutions = resolve_stt_options(
        requested=SttOptions(temperature=0.2, vad=False),
        preset=SttOptions(beam_size=3, temperature=0.5),
        defaults=ResolvedSttOptions(),
        capabilities=FULL_CAPABILITIES,
        backend=SttBackendName.CTRANSLATE2,
    )
    assert (resolved.beam_size, resolved.temperature, resolved.vad.enabled) == (3, 0.2, False)
    assert substitutions == ()


def test_model_resolution() -> None:
    """Fail for missing caller models and substitute missing preset models."""
    assert resolve_stt_model(requested="small", preset_model=None, default="small", installed={"small"}) == (
        "small",
        (),
    )
    model, subs = resolve_stt_model(requested=None, preset_model="medium", default="small", installed={"small"})
    assert model == "small"
    assert subs[0].option == "model"
    with pytest.raises(ModelNotInstalledError):
        resolve_stt_model(requested="large-v3", preset_model=None, default="small", installed={"small"})
    with pytest.raises(ModelNotInstalledError):
        resolve_stt_model(requested=None, preset_model=None, default="small", installed=set())


def test_tts_options_layering_and_validation() -> None:
    """Merge TTS options and reject invalid formats."""
    resolved = resolve_tts_options(
        requested=TtsOptions(output_format="mp3"), preset=TtsOptions(gap_ms=50), defaults=ResolvedTtsOptions()
    )
    assert (resolved.output_format, resolved.gap_ms) == ("mp3", 50)
    with pytest.raises(InvalidRequestError):
        resolve_tts_options(requested=TtsOptions(output_format="aac"), preset=None, defaults=ResolvedTtsOptions())


def test_platform_profile_follows_selected_devices() -> None:
    """Report linux-cuda only when a component actually runs on CUDA."""
    assert platform_profile(cuda_host(), devices=("cpu", "cpu", None)) == "linux-cpu"
    assert platform_profile(cuda_host(), devices=("cuda", None, None)) == "linux-cuda"
    assert platform_profile(mac_host(), devices=("metal", "metal", "cpu")) == "macos-native"
