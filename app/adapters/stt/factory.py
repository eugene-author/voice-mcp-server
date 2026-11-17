"""Builds the selected STT backend; backend modules are imported lazily so unused runtimes are not loaded."""

from app.core.settings import Settings
from app.domain.host import HostProfile
from app.domain.stt import SttBackendName, SttRuntime
from app.ports.stt import SttBackend


def build_stt_backend(name: SttBackendName, runtime: SttRuntime, settings: Settings, host: HostProfile) -> SttBackend:
    """Return the backend for name; mlx-whisper is imported only on macOS hosts with the mlx extra."""
    if name is SttBackendName.MLX:
        from app.adapters.stt.mlx_whisper.backend import MlxWhisperBackend

        return MlxWhisperBackend()
    from app.adapters.stt.ctranslate2.backend import CTranslate2Backend

    return CTranslate2Backend(runtime, cpu_threads=host.cpu_count if runtime.device == "cpu" else 0)
