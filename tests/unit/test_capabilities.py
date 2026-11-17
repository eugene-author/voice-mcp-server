"""HostCapabilities rules for missing TTS and denoise runtimes."""

from pathlib import Path

import pytest

from app.domain.errors import CapabilityUnavailableError
from app.domain.jobs import RegisterVoiceCommand, SynthesizeCommand, TranscribeCommand
from app.domain.stt import SttOptions
from app.services.capabilities import HostCapabilities
from tests.mocks.stt import MockSttBackend
from tests.mocks.tts import MockTtsEngine

STT = MockSttBackend().capabilities
TTS = MockTtsEngine().capabilities


def test_missing_tts_makes_tts_commands_unavailable() -> None:
    """Name the missing extra for TTS and voice commands only."""
    capabilities = HostCapabilities(stt=MockSttBackend().capabilities, tts=None, denoise=False)
    for command in (
        SynthesizeCommand(text="hi", voice_id="v"),
        RegisterVoiceCommand(name="n", reference_audio_path=Path("/x.wav"), consent_confirmed=True),
    ):
        with pytest.raises(CapabilityUnavailableError, match="'tts' extra"):
            capabilities.require_for(command)
    capabilities.require_for(TranscribeCommand(input_path=Path("/x.wav")))


def test_explicit_denoise_requires_denoiser() -> None:
    """Reject explicit denoise requests without DeepFilterNet."""
    without = HostCapabilities(stt=STT, tts=TTS, denoise=False)
    with pytest.raises(CapabilityUnavailableError, match="denoise"):
        without.require_for(TranscribeCommand(input_path=Path("/x.wav"), options=SttOptions(denoise=True)))
    with_denoiser = HostCapabilities(stt=STT, tts=TTS, denoise=True)
    with_denoiser.require_for(TranscribeCommand(input_path=Path("/x.wav"), options=SttOptions(denoise=True)))
