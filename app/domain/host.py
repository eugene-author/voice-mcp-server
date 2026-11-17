"""Host hardware and installed-extra facts."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """NVIDIA device reported for information only."""

    name: str
    memory_mb: int


@dataclass(frozen=True, slots=True)
class HostProfile:
    """Hardware and installed-extra facts probed once at startup."""

    system: str
    """Value of platform.system()."""

    machine: str
    """Value of platform.machine()."""

    cpu_count: int
    total_ram_mb: int
    mlx_available: bool
    mlx_audio_available: bool
    qwen_tts_available: bool
    denoise_runtime_available: bool
    """Whether onnxruntime is installed for DeepFilterNet3."""

    torch_cuda_available: bool
    """PyTorch CUDA availability; used for TTS and denoise devices only, not for STT."""

    cuda_device_count: int
    """CTranslate2 CUDA device count."""

    cuda_compute_types: frozenset[str]
    """CTranslate2 compute types supported on CUDA device 0."""

    gpus: tuple[GpuDevice, ...] = ()

    @property
    def is_apple_silicon(self) -> bool:
        """Return whether the host is macOS on arm64."""
        return self.system == "Darwin" and self.machine == "arm64"

    @property
    def cuda_memory_mb(self) -> int | None:
        """Return the memory of the first reported GPU, if any."""
        return self.gpus[0].memory_mb if self.gpus else None
