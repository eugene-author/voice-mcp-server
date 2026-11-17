"""Builds the standalone Silero detector; silero-vad is imported lazily because it ships with the mlx extra."""

from app.ports.vad import VoiceActivityDetector


def build_vad() -> VoiceActivityDetector:
    """Return the Silero detector used on the mlx STT path."""
    from app.adapters.vad.silero.detector import SileroDetector

    return SileroDetector()
