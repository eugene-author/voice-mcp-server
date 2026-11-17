"""Long-audio stress: many minutes of speech and silence keep ordered, source-relative timestamps."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.adapters.stt.ctranslate2.backend import CTranslate2Backend
from app.domain.models import InstalledModel, ModelKind
from app.domain.stt import SttBackendName, SttRequest, SttRuntime, VadSettings

pytestmark = pytest.mark.models
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "var" / "models" / "stt" / "ctranslate2" / "small"
FIXTURE = ROOT / "tests" / "fixtures" / "demo" / "jessica.wav"
REPEATS = 20
GAP_SECONDS = 4.0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="needs the small CTranslate2 model")
def test_long_audio_keeps_ordered_source_timestamps(tmp_path: Path) -> None:
    """Transcribe about three minutes of repeated speech separated by silence."""
    speech, rate = sf.read(str(FIXTURE), dtype="float32")
    gap = np.zeros(int(GAP_SECONDS * rate), dtype=np.float32)
    audio = np.concatenate([part for _ in range(REPEATS) for part in (speech, gap)])
    path = tmp_path / "long.wav"
    sf.write(str(path), audio, rate)
    backend = CTranslate2Backend(SttRuntime(SttBackendName.CTRANSLATE2, "cpu", "int8"))
    backend.load(InstalledModel(name="small", kind=ModelKind.STT, backend="ctranslate2", local_path=MODEL_DIR))
    result = backend.transcribe(
        SttRequest(
            audio_path=path, language="en", beam_size=1, temperature=0.0, word_timestamps=True, vad=VadSettings()
        )
    )
    duration = len(audio) / rate
    starts = [segment.start for segment in result.segments]
    assert result.duration_seconds == pytest.approx(duration, abs=0.1)
    assert starts == sorted(starts)
    assert result.segments[-1].end <= duration + 0.5
    assert result.text.lower().count("nice to meet you") >= REPEATS - 2
