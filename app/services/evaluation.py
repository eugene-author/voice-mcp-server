"""Offline WER/CER evaluation over a directory of audio files with ground-truth transcripts."""

import asyncio
import time
from pathlib import Path
from typing import Any

from app.domain.errors import InvalidRequestError
from app.domain.evaluation import char_error_rate, word_error_rate
from app.domain.stt import SttRequest, VadSettings
from app.ports.media import MediaConverter
from app.services.model_host import ModelHost

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm")
EVAL_WAV_FILE = "eval-{stem}.wav"


def _dataset_pairs(dataset: Path) -> list[tuple[Path, Path]]:
    """Return (audio, transcript) pairs found in the dataset directory."""
    return sorted(
        (audio, audio.with_suffix(".txt"))
        for audio in dataset.iterdir()
        if audio.suffix.lower() in AUDIO_SUFFIXES and audio.with_suffix(".txt").is_file()
    )


async def evaluate_dataset(
    dataset: Path,
    *,
    model_host: ModelHost,
    converter: MediaConverter,
    model: str,
    language: str | None,
    workspace: Path,
) -> dict[str, Any]:
    """Transcribe every audio file that has a sibling .txt transcript and report per-file and mean error rates."""
    pairs = await asyncio.to_thread(_dataset_pairs, dataset)
    if not pairs:
        raise InvalidRequestError("dataset has no audio files with matching .txt transcripts")
    installed = model_host.stt_model(model)
    capabilities = model_host.stt.capabilities
    items: list[dict[str, Any]] = []
    for audio, truth in pairs:
        wav = await converter.to_wav(audio, workspace / EVAL_WAV_FILE.format(stem=audio.stem), sample_rate=16000)
        started = time.perf_counter()
        result = await model_host.transcribe(
            installed,
            SttRequest(
                audio_path=wav,
                language=language,
                beam_size=5 if capabilities.beam_search else 1,
                temperature=0.0,
                word_timestamps=False,
                vad=VadSettings(enabled=capabilities.native_vad),
            ),
        )
        elapsed = time.perf_counter() - started
        wav.unlink(missing_ok=True)
        reference = await asyncio.to_thread(truth.read_text, encoding="utf-8")
        items.append(
            {
                "file": audio.name,
                "wer": round(word_error_rate(reference, result.text), 4),
                "cer": round(char_error_rate(reference, result.text), 4),
                "rtf": round(elapsed / result.duration_seconds, 4) if result.duration_seconds else None,
                "hypothesis": result.text,
            }
        )
    return {
        "model": model,
        "backend": str(model_host.stt.name),
        "files": len(items),
        "mean_wer": round(sum(item["wer"] for item in items) / len(items), 4),
        "mean_cer": round(sum(item["cer"] for item in items) / len(items), 4),
        "items": items,
    }
