"""Literal host profiles for selection tests."""

from app.domain.host import GpuDevice, HostProfile


def host_profile(**overrides: object) -> HostProfile:
    """Return a Linux CPU host profile with optional field overrides."""
    values: dict[str, object] = {
        "system": "Linux",
        "machine": "x86_64",
        "cpu_count": 8,
        "total_ram_mb": 32768,
        "mlx_available": False,
        "mlx_audio_available": False,
        "qwen_tts_available": True,
        "denoise_runtime_available": False,
        "torch_cuda_available": False,
        "cuda_device_count": 0,
        "cuda_compute_types": frozenset(),
        "gpus": (),
    }
    values.update(overrides)
    return HostProfile(**values)  # type: ignore[arg-type]


def cuda_host(
    memory_mb: int = 24576, compute_types: frozenset[str] = frozenset({"float16", "int8_float16", "float32", "int8"})
) -> HostProfile:
    """Return a Linux host with one NVIDIA GPU."""
    return host_profile(
        cuda_device_count=1,
        cuda_compute_types=compute_types,
        torch_cuda_available=True,
        gpus=(GpuDevice(name="NVIDIA Test", memory_mb=memory_mb),),
    )


def mac_host(*, mlx: bool = True, mlx_audio: bool = True, qwen_tts: bool = False) -> HostProfile:
    """Return a macOS Apple Silicon host."""
    return host_profile(
        system="Darwin", machine="arm64", mlx_available=mlx, mlx_audio_available=mlx_audio, qwen_tts_available=qwen_tts
    )
