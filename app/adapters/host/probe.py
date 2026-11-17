"""Host probing: the only code that inspects the platform, CUDA and installed extras."""

import getpass
import importlib.util
import logging
import os
import platform
import shutil
import subprocess

from app.domain.host import GpuDevice, HostProfile

logger = logging.getLogger(__name__)
NVIDIA_SMI_TIMEOUT_SECONDS = 5


def _module_available(name: str) -> bool:
    """Return whether a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError, ValueError:
        return False


def _total_ram_mb() -> int:
    """Return physical memory in MiB, or 0 when unknown."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except ValueError, OSError, AttributeError:
        return 0


def _ct2_cuda() -> tuple[int, frozenset[str]]:
    """Return CTranslate2's CUDA device count and device 0 compute types; CTranslate2 is a base dependency."""
    import ctranslate2

    try:
        count = ctranslate2.get_cuda_device_count()
        types = frozenset(ctranslate2.get_supported_compute_types("cuda")) if count > 0 else frozenset()
    except (RuntimeError, ValueError) as exc:
        logger.warning("CTranslate2 CUDA probe failed", extra={"error": str(exc)})
        return 0, frozenset()
    return count, types


def _nvidia_gpus() -> tuple[GpuDevice, ...]:
    """Return NVIDIA device names and memory from nvidia-smi, for reporting only."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    try:
        completed = subprocess.run(  # noqa: S603
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
            check=True,
        )
    except OSError, subprocess.SubprocessError:
        return ()
    devices: list[GpuDevice] = []
    for line in completed.stdout.strip().splitlines():
        name, _, memory = line.rpartition(",")
        if name and memory.strip().isdigit():
            devices.append(GpuDevice(name=name.strip(), memory_mb=int(memory.strip())))
    return tuple(devices)


def _torch_cuda_available() -> bool:
    """Return PyTorch CUDA availability; torch is imported lazily because it is an optional, heavy extra."""
    if not _module_available("torch"):
        return False
    import torch

    return bool(torch.cuda.is_available())


class SystemHostProbe:
    """Probes the real host."""

    def probe(self) -> HostProfile:
        """Return the host profile."""
        cuda_count, cuda_types = _ct2_cuda()
        return HostProfile(
            system=platform.system(),
            machine=platform.machine(),
            cpu_count=os.cpu_count() or 1,
            total_ram_mb=_total_ram_mb(),
            mlx_available=_module_available("mlx_whisper"),
            mlx_audio_available=_module_available("mlx_audio"),
            qwen_tts_available=_module_available("qwen_tts"),
            denoise_runtime_available=_module_available("onnxruntime"),
            torch_cuda_available=_torch_cuda_available(),
            cuda_device_count=cuda_count,
            cuda_compute_types=cuda_types,
            gpus=_nvidia_gpus(),
        )

    def probe_basic(self) -> HostProfile:
        """Return OS, CPU and memory facts only, importing no inference library, for the api role."""
        return HostProfile(
            system=platform.system(),
            machine=platform.machine(),
            cpu_count=os.cpu_count() or 1,
            total_ram_mb=_total_ram_mb(),
            mlx_available=False,
            mlx_audio_available=False,
            qwen_tts_available=False,
            denoise_runtime_available=False,
            torch_cuda_available=False,
            cuda_device_count=0,
            cuda_compute_types=frozenset(),
        )


def os_user() -> str:
    """Return the OS user name used as the trusted-local stdio subject."""
    try:
        return getpass.getuser()
    except KeyError, OSError:
        return f"uid-{os.getuid()}"
