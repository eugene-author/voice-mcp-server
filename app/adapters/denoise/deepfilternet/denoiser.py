"""DeepFilterNet3 denoising with the full-pipeline ONNX export (torchDF) on ONNX Runtime."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import soundfile as sf

from app.domain.errors import InferenceError, MediaError
from app.domain.models import InstalledModel

MODEL_FILE = "denoiser_model.onnx"
SAMPLE_RATE = 48000
FFT_SIZE = 960
LOOKAHEAD_FRAMES = 2
"""DeepFilterNet3 looks two frames ahead; a streaming graph emits that much later, so it is compensated here."""

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class DeepFilterNetDenoiser:
    """Denoiser that streams 48 kHz audio through DeepFilterNet3 frame by frame, carrying the model state."""

    def __init__(self, *, device: str, attenuation_limit_db: float = 0.0) -> None:
        """Start unloaded; attenuation_limit_db 0 means full suppression, larger values keep some noise."""
        self._requested_device = device
        self._attenuation_limit = np.asarray(attenuation_limit_db, dtype=np.float32)
        self._session: ort.InferenceSession | None = None
        self._frame_size = 0
        self._state_size = 0

    @property
    def device(self) -> str:
        """Return cuda when the CUDA execution provider is in use or will be used, otherwise cpu."""
        if self._session is not None:
            return "cuda" if CUDA_PROVIDER in self._session.get_providers() else "cpu"
        return "cuda" if self._use_cuda() else "cpu"

    def load(self, model: InstalledModel) -> None:
        """Load the ONNX graph from the installed directory and read its frame and state sizes."""
        path = model.local_path / MODEL_FILE
        providers = [CUDA_PROVIDER, CPU_PROVIDER] if self._use_cuda() else [CPU_PROVIDER]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        except (RuntimeError, OSError, ValueError) as exc:
            raise InferenceError("could not load the DeepFilterNet model") from exc
        shapes = {item.name: item.shape for item in session.get_inputs()}
        try:
            self._frame_size, self._state_size = int(shapes["input_frame"][0]), int(shapes["states"][0])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InferenceError("the DeepFilterNet model has an unexpected interface") from exc
        self._session = session

    def denoise(self, source: Path, destination: Path) -> None:
        """Write a denoised copy of a 48 kHz mono WAV with the same length and alignment."""
        samples, rate = sf.read(str(source), dtype="float32", always_2d=False)
        if rate != SAMPLE_RATE or np.ndim(samples) != 1:
            raise MediaError(f"denoise input must be {SAMPLE_RATE} Hz mono")
        sf.write(str(destination), self.enhance(np.asarray(samples, dtype=np.float32)), SAMPLE_RATE, subtype="FLOAT")

    def enhance(self, samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Return enhanced samples aligned with the input: STFT and lookahead delays are padded and trimmed."""
        session = self._require_session()
        hop = self._frame_size
        length = len(samples)
        delay = FFT_SIZE - hop + LOOKAHEAD_FRAMES * hop
        padded_length = length + (-length) % hop
        padded = np.zeros(padded_length + delay + hop, dtype=np.float32)
        padded[:length] = samples
        states = np.zeros(self._state_size, dtype=np.float32)
        frames: list[npt.NDArray[np.float32]] = []
        try:
            for start in range(0, len(padded), hop):
                enhanced, states, _ = session.run(
                    None,
                    {
                        "input_frame": padded[start : start + hop],
                        "states": states,
                        "atten_lim_db": self._attenuation_limit,
                    },
                )
                frames.append(enhanced)
        except (RuntimeError, ValueError) as exc:
            raise InferenceError("DeepFilterNet inference failed", retryable=True) from exc
        output: npt.NDArray[np.float32] = np.concatenate(frames).astype(np.float32, copy=False)
        return output[delay : delay + length]

    def unload(self) -> None:
        """Release the session."""
        self._session = None

    def _use_cuda(self) -> bool:
        """Return whether CUDA was requested and this onnxruntime build provides it."""
        return self._requested_device == "cuda" and CUDA_PROVIDER in ort.get_available_providers()

    def _require_session(self) -> ort.InferenceSession:
        """Return the loaded session or raise InferenceError."""
        if self._session is None:
            raise InferenceError("the DeepFilterNet model is not loaded")
        return self._session
