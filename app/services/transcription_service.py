"""Speech-to-text pipeline from probe to publish; each stage ends in a committed checkpoint for retries."""

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.domain.assets import AssetRole, UploadedInput, job_input_key, job_output_key
from app.domain.errors import MediaError
from app.domain.jobs import ClaimedJob, ExecutionInfo, JobCompletion, JobPhase, ProgressCallback, TranscriptionSpec
from app.domain.quality import QualityReport, annotate, fillers_for, flag_segments, remove_fillers, remove_repetitions
from app.domain.stt import SttRequest, TranscriptionResult
from app.domain.subtitles import render_json, render_srt, render_txt, render_vtt, transcript_from_document
from app.domain.timeline import SpeechTimeline, map_to_source
from app.ports.media import MediaConverter, MediaProbe
from app.services.inference_progress import InferenceProgress
from app.services.input_transfer import fetch_input
from app.services.job_workspace import JobWorkspace
from app.services.model_host import ModelHost
from app.services.publishing import AssetPublisher, StagedFile
from app.services.staged_job_assets import StagedJobAssets

logger = logging.getLogger(__name__)
WORKING_SAMPLE_RATE = 16000
INPUT_WAV_FILE = "input.wav"
SPEECH_WAV_FILE = "speech.wav"
INPUT_48K_WAV_FILE = "input_48k.wav"
DENOISED_48K_WAV_FILE = "denoised_48k.wav"
DENOISED_WAV_FILE = "denoised.wav"
TRANSCRIPT_FILES = (
    ("txt", "transcript.txt"),
    ("json", "transcript.json"),
    ("srt", "transcript.srt"),
    ("vtt", "transcript.vtt"),
)
TRANSCRIPT_FILENAMES = tuple(filename for _, filename in TRANSCRIPT_FILES)


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    """Everything a transcription retry needs to publish and complete without running inference again."""

    result: TranscriptionResult
    warnings: tuple[str, ...]
    metrics: dict[str, Any]
    execution: ExecutionInfo

    @classmethod
    def parse(cls, directory: Path, metadata: Mapping[str, Any]) -> _Checkpoint:
        """Rebuild the checkpoint from transcript.json and outcome.json; raise on missing or malformed fields."""
        document = json.loads((directory / "transcript.json").read_text(encoding="utf-8"))
        warnings, metrics = metadata["warnings"], metadata["metrics"]
        if not isinstance(warnings, list) or not isinstance(metrics, dict):
            raise TypeError("warnings must be a list and metrics an object")
        return cls(
            result=transcript_from_document(document),
            warnings=tuple(str(item) for item in warnings),
            metrics=metrics,
            execution=ExecutionInfo(**metadata["execution"]),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return the outcome.json content."""
        execution = self.execution
        return {
            "warnings": list(self.warnings),
            "metrics": self.metrics,
            "execution": {
                "backend": execution.backend,
                "model_id": execution.model_id,
                "device": execution.device,
                "compute_type": execution.compute_type,
                "model_revision": execution.model_revision,
            },
        }


@dataclass(frozen=True, slots=True)
class _Prepared:
    """Audio ready for the STT backend plus the timeline needed to map times back."""

    audio_path: Path
    timeline: SpeechTimeline | None
    warnings: tuple[str, ...]


class TranscriptionService:
    """Runs one speech_to_text job end to end and returns its completion."""

    def __init__(
        self,
        *,
        probe: MediaProbe,
        converter: MediaConverter,
        model_host: ModelHost,
        publisher: AssetPublisher,
        staged: StagedJobAssets,
        workspace: JobWorkspace,
        fillers: Mapping[str, Sequence[str]],
        max_duration_seconds: float,
        text_preview_chars: int,
        inference_progress: InferenceProgress,
    ) -> None:
        """Store injected collaborators and limits."""
        self._probe = probe
        self._inference_progress = inference_progress
        self._converter = converter
        self._model_host = model_host
        self._publisher = publisher
        self._staged = staged
        self._workspace = workspace
        self._fillers = fillers
        self._max_duration_seconds = max_duration_seconds
        self._text_preview_chars = text_preview_chars

    async def execute(self, job: ClaimedJob, progress: ProgressCallback) -> JobCompletion:
        """Transcribe, export and publish; a retry after a publish failure reuses a complete staged checkpoint."""
        spec = TranscriptionSpec.from_payload(job.input_payload)
        checkpoint = self._staged.load(job, required=TRANSCRIPT_FILENAMES, parse=_Checkpoint.parse)
        if checkpoint is None:
            checkpoint = await self._transcribe(job, spec, progress)
        await progress(JobPhase.PUBLISHING, 92)
        uploaded = UploadedInput.from_payload(job.input_payload)
        assets = await self._publisher.publish(self._staged_files(job, spec, uploaded))
        if uploaded is not None and spec.preserve_input:
            assets = (*assets, uploaded.as_original())
        result = checkpoint.result
        return JobCompletion(
            result={
                "text": result.text[: self._text_preview_chars],
                "language": result.language,
                "duration_seconds": round(result.duration_seconds, 3),
            },
            assets=assets,
            warnings=checkpoint.warnings,
            execution=checkpoint.execution,
            transcript=result,
            metrics=checkpoint.metrics,
        )

    async def _transcribe(self, job: ClaimedJob, spec: TranscriptionSpec, progress: ProgressCallback) -> _Checkpoint:
        """Run every stage up to and including export into a committed staged checkpoint."""
        started = time.perf_counter()
        uploaded = UploadedInput.from_payload(job.input_payload)
        source = await fetch_input(self._publisher, uploaded, spec.input_path, self._workspace.claim_dir(job))
        await progress(JobPhase.DECODING, 5)
        info = await self._probe.probe(source)
        if info.duration_seconds > self._max_duration_seconds:
            raise MediaError(f"audio is longer than {self._max_duration_seconds:.0f} seconds")
        prepared = await self._prepare(job, spec, source, progress)
        await progress(JobPhase.INFERENCING, 30)
        model = self._model_host.stt_model(spec.model)
        backend = self._model_host.stt
        inference_seconds = 0.0
        if prepared.timeline is not None and prepared.timeline.is_empty:
            result = TranscriptionResult(
                segments=(),
                language=spec.language or "",
                language_probability=None,
                duration_seconds=info.duration_seconds,
            )
        else:
            request = self._request(spec, prepared)
            inference_started = time.perf_counter()
            async with self._inference_progress.track(
                progress,
                phase=JobPhase.INFERENCING,
                start=30,
                end=85,
                units=info.duration_seconds,
                key=model.name,
                learn=self._model_host.loaded_models["stt"] == model.name,
            ):
                result = await self._model_host.transcribe(model, request)
            inference_seconds = time.perf_counter() - inference_started
            if prepared.timeline is not None:
                result = replace(map_to_source(result, prepared.timeline), duration_seconds=info.duration_seconds)
        await progress(JobPhase.EXPORTING, 85)
        result, report = self._quality(result, spec)
        warnings = (*prepared.warnings, *report.warnings())
        elapsed = time.perf_counter() - started
        runtime = backend.runtime
        execution = ExecutionInfo(
            backend=str(backend.name),
            model_id=model.name,
            device=runtime.device,
            compute_type=runtime.compute_type,
            model_revision=model.revision,
        )
        metrics: dict[str, Any] = {
            "rtf": round(elapsed / info.duration_seconds, 4) if info.duration_seconds else None,
            "processing_seconds": round(elapsed, 3),
            "inference_seconds": round(inference_seconds, 3),
            "audio_seconds": round(info.duration_seconds, 3),
            "segments": len(result.segments),
            "denoise": spec.options.denoise,
            "quality": report.to_dict(),
            "wer": None,
            "cer": None,
        }
        checkpoint = _Checkpoint(result=result, warnings=warnings, metrics=metrics, execution=execution)
        self._export(job, checkpoint, report)
        return checkpoint

    async def _prepare(
        self, job: ClaimedJob, spec: TranscriptionSpec, source: Path, progress: ProgressCallback
    ) -> _Prepared:
        """Decode the local source to 16 kHz mono WAV, then optionally denoise and run standalone VAD."""
        work = self._workspace.claim_dir(job)
        audio = await self._converter.to_wav(source, work / INPUT_WAV_FILE, sample_rate=WORKING_SAMPLE_RATE)
        warnings: list[str] = []
        if spec.options.denoise:
            await progress(JobPhase.DENOISING, 12)
            audio = await self._denoise(audio, work)
        if self._model_host.stt.capabilities.native_vad or not spec.options.vad.enabled:
            return _Prepared(audio_path=audio, timeline=None, warnings=tuple(warnings))
        await progress(JobPhase.VAD, 20)
        samples, rate = sf.read(str(audio), dtype="float32")
        regions = await self._model_host.detect_speech(np.asarray(samples, dtype=np.float32), rate, spec.options.vad)
        timeline = SpeechTimeline(regions, merge_gap_seconds=spec.options.vad.merge_gap_ms / 1000)
        if timeline.is_empty:
            warnings.append("no_speech_detected")
            return _Prepared(audio_path=audio, timeline=timeline, warnings=tuple(warnings))
        speech_path = work / SPEECH_WAV_FILE
        sf.write(str(speech_path), timeline.extract(np.asarray(samples, dtype=np.float32), rate), rate)
        return _Prepared(audio_path=speech_path, timeline=timeline, warnings=tuple(warnings))

    async def _denoise(self, audio: Path, work: Path) -> Path:
        """Denoise at DeepFilterNet's rate and convert back to the working rate."""
        denoised = work / DENOISED_48K_WAV_FILE
        upsampled = await self._converter.to_wav(audio, work / INPUT_48K_WAV_FILE, sample_rate=48000)
        await self._model_host.denoise(upsampled, denoised)
        return await self._converter.to_wav(denoised, work / DENOISED_WAV_FILE, sample_rate=WORKING_SAMPLE_RATE)

    @staticmethod
    def _request(spec: TranscriptionSpec, prepared: _Prepared) -> SttRequest:
        """Build the backend request; standalone VAD already removed silence on the mlx path."""
        options = spec.options
        vad = options.vad if prepared.timeline is None else replace(options.vad, enabled=False)
        return SttRequest(
            audio_path=prepared.audio_path,
            language=spec.language,
            beam_size=options.beam_size,
            temperature=options.temperature,
            word_timestamps=options.word_timestamps,
            vad=vad,
            initial_prompt=options.initial_prompt,
            hotwords=options.hotwords,
            condition_on_previous_text=options.condition_on_previous_text,
            no_speech_threshold=options.no_speech_threshold,
        )

    def _quality(
        self, result: TranscriptionResult, spec: TranscriptionSpec
    ) -> tuple[TranscriptionResult, QualityReport]:
        """Flag heuristics, annotate fillers and repeats, and apply explicitly requested removals."""
        options = spec.options
        fillers = fillers_for(result.language, self._fillers)
        report = QualityReport(
            flags=flag_segments(result.segments) if options.segment_flags else (),
            annotations=annotate(result.segments, fillers),
        )
        if options.remove_fillers:
            result = remove_fillers(result, fillers)
        if options.remove_repetitions:
            result = remove_repetitions(result)
        return result, report

    def _export(self, job: ClaimedJob, checkpoint: _Checkpoint, report: QualityReport) -> None:
        """Write the transcript files, then commit the checkpoint metadata and marker."""
        result = checkpoint.result
        contents = {
            "transcript.txt": render_txt(result),
            "transcript.json": render_json(result, extra={"quality": report.to_dict()}),
            "transcript.srt": render_srt(result),
            "transcript.vtt": render_vtt(result),
        }
        for filename, text in contents.items():
            self._staged.write_text(job, filename, text)
        self._staged.commit(job, checkpoint.to_metadata(), required=TRANSCRIPT_FILENAMES)

    def _staged_files(
        self, job: ClaimedJob, spec: TranscriptionSpec, uploaded: UploadedInput | None
    ) -> list[StagedFile]:
        """Return the transcript outputs and, when enabled and not already uploaded, the original to publish."""
        files = [
            StagedFile(
                name=name,
                path=self._staged.path(job, filename),
                role=AssetRole.OUTPUT,
                key=job_output_key(job.owner, job.id, filename, claim_token=job.claim_token),
            )
            for name, filename in TRANSCRIPT_FILES
        ]
        if spec.preserve_input and uploaded is None:
            source = Path(spec.input_path)
            extension = source.suffix.lstrip(".") or "bin"
            key = job_input_key(job.owner, job.id, extension, claim_token=job.claim_token)
            files.append(StagedFile(name="original", path=source, role=AssetRole.INPUT, key=key))
        return files
