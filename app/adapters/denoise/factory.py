"""Builds the DeepFilterNet denoiser; onnxruntime is imported lazily because the denoise extra is optional."""

from typing import Literal

from app.ports.denoise import Denoiser


def build_denoiser(*, device: Literal["cuda", "cpu"], attenuation_limit_db: float) -> Denoiser:
    """Return the DeepFilterNet3 ONNX denoiser for the requested device."""
    from app.adapters.denoise.deepfilternet.denoiser import DeepFilterNetDenoiser

    return DeepFilterNetDenoiser(device=device, attenuation_limit_db=attenuation_limit_db)
