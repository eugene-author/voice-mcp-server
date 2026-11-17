"""Text-to-speech pipeline from chunking to publish; the encoded output is checkpointed so retries skip synthesis."""

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.domain.assets import AssetLocation, AssetRole, StorageBackend, job_output_key
from app.domain.audio import peak, rms, stitch
from app.domain.errors import InferenceError, InvalidRequestError, NotFoundError
from app.domain.jobs import ClaimedJob, ExecutionInfo, JobCompletion, JobPhase, ProgressCallback, SynthesisSpec
from app.domain.models import InstalledModel
from app.domain.speech_text import chunk_text
from app.domain.tts import CloneReference, PresetSpeaker, SynthesisRequest
from app.ports.media import MediaConverter
from app.ports.storage import AssetStore
from app.ports.unit_of_work import UnitOfWorkFactory
from app.services.inference_progress import InferenceProgress
from app.services.job_workspace import JobWorkspace
from app.services.model_host import ModelHost
from app.services.publishing import AssetPublisher, StagedFile
from app.services.staged_job_assets import StagedJobAssets

logger = logging.getLogger(__name__)
SILENCE_RMS = 1e-4
STITCHED_WAV_FILE = "stitched.wav"
MASTERED_WAV_FILE = "mastered.wav"
DEFAULT_REFERENCE_SUFFIX = ".wav"


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    """Metadata of a staged synthesis output, enough to publish and complete without synthesizing again."""

    sample_rate: int
    warnings: tuple[str, ...]
    execution: ExecutionInfo
    metrics: dict[str, Any]

    @property
    def output_seconds(self) -> float:
        """Return the duration of the staged audio."""
        return float(self.metrics["output_seconds"])

    @classmethod
    def parse(cls, _directory: Path, metadata: Mapping[str, Any]) -> _Checkpoint:
        """Rebuild the checkpoint from outcome.json; raise on missing or malformed fields."""
        warnings, metrics = metadata["warnings"], metadata["metrics"]
        if not isinstance(warnings, list) or not isinstance(metrics, dict):
            raise TypeError("warnings must be a list and metrics an object")
        checkpoint = cls(
            sample_rate=int(metadata["sample_rate"]),
            warnings=tuple(str(item) for item in warnings),
            execution=ExecutionInfo(**metadata["execution"]),
            metrics=metrics,
        )
        if checkpoint.sample_rate <= 0 or checkpoint.output_seconds < 0:
            raise ValueError("invalid sample rate or duration")
        return checkpoint

    def to_metadata(self) -> dict[str, Any]:
        """Return the outcome.json content."""
        execution = self.execution
        return {
            "sample_rate": self.sample_rate,
            "warnings": list(self.warnings),
            "execution": {
                "backend": execution.backend,
                "model_id": execution.model_id,
                "device": execution.device,
                "compute_type": execution.compute_type,
                "model_revision": execution.model_revision,
            },
            "metrics": self.metrics,
        }


class SynthesisService:
    """Runs one text_to_speech job; adapters only turn one text chunk into samples."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkFactory,
        store: AssetStore,
        converter: MediaConverter,
        model_host: ModelHost,
        publisher: AssetPublisher,
        staged: StagedJobAssets,
        workspace: JobWorkspace,
        cloning_checkpoint: str,
        preset_checkpoint: str | None,
        inference_progress: InferenceProgress,
    ) -> None:
        """Store injected collaborators and checkpoint names."""
        self._uow = uow
        self._store = store
        self._converter = converter
        self._model_host = model_host
        self._publisher = publisher
        self._staged = staged
        self._workspace = workspace
        self._cloning_checkpoint = cloning_checkpoint
        self._inference_progress = inference_progress
        self._preset_checkpoint = preset_checkpoint

    async def execute(self, job: ClaimedJob, progress: ProgressCallback) -> JobCompletion:
        """Synthesize and publish; a retry after a publish failure reuses a complete staged checkpoint."""
        spec = SynthesisSpec.from_payload(job.input_payload)
        output_name = f"speech.{spec.options.output_format}"
        checkpoint = self._staged.load(job, required=(output_name,), parse=_Checkpoint.parse)
        if checkpoint is None:
            checkpoint = await self._synthesize(job, spec, output_name, progress)
        await progress(JobPhase.PUBLISHING, 92)
        assets = await self._publisher.publish(
            [
                StagedFile(
                    name=spec.options.output_format,
                    path=self._staged.path(job, output_name),
                    role=AssetRole.OUTPUT,
                    key=job_output_key(job.owner, job.id, output_name, claim_token=job.claim_token),
                    duration_seconds=checkpoint.output_seconds,
                    sample_rate=checkpoint.sample_rate,
                )
            ]
        )
        return JobCompletion(
            result={"voice_id": spec.voice_id, "duration_seconds": checkpoint.output_seconds},
            assets=assets,
            warnings=checkpoint.warnings,
            execution=checkpoint.execution,
            metrics=checkpoint.metrics,
        )

    async def _synthesize(
        self, job: ClaimedJob, spec: SynthesisSpec, output_name: str, progress: ProgressCallback
    ) -> _Checkpoint:
        """Run synthesis, encode the output into the staged directory and commit the checkpoint."""
        started = time.perf_counter()
        work = self._workspace.claim_dir(job)
        voice, model = await self._voice(spec, work)
        chunks = chunk_text(spec.text, max_chars=spec.options.max_chunk_chars)
        if not chunks:
            raise InvalidRequestError("text contains nothing to speak")
        tts = self._model_host.tts
        if tts is None:
            raise InferenceError("no TTS runtime is available")
        rendered: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        sample_rate = 0
        generation_seconds = 0.0
        for index, chunk in enumerate(chunks):
            start = 10 + round(70 * index / len(chunks))
            end = 10 + round(70 * (index + 1) / len(chunks))
            await progress(JobPhase.INFERENCING, start)
            chunk_started = time.perf_counter()
            async with self._inference_progress.track(
                progress,
                phase=JobPhase.INFERENCING,
                start=start,
                end=end,
                units=len(chunk),
                key=model.name,
                learn=self._model_host.loaded_models["tts"] == model.name,
            ):
                audio = await self._model_host.synthesize(
                    model, SynthesisRequest(text=chunk, language=spec.language, voice=voice)
                )
            generation_seconds += time.perf_counter() - chunk_started
            if rms(audio.samples) < SILENCE_RMS:
                raise InferenceError(f"chunk {index + 1} of {len(chunks)} produced silence", retryable=True)
            if sample_rate and audio.sample_rate != sample_rate:
                raise InferenceError("chunks were produced at different sample rates")
            sample_rate = audio.sample_rate
            rendered.append(audio.samples)
        await progress(JobPhase.EXPORTING, 82)
        stitched = stitch(rendered, sample_rate=sample_rate, gap_ms=spec.options.gap_ms)
        raw = work / STITCHED_WAV_FILE
        sf.write(str(raw), stitched, sample_rate, subtype="FLOAT")
        mastered = raw
        if spec.options.loudness_normalization:
            mastered = await self._converter.normalize_loudness(raw, work / MASTERED_WAV_FILE)
        output = self._staged.path(job, output_name)
        partial = output.with_name(output.name + ".part")
        await self._converter.encode(mastered, partial, output_format=spec.options.output_format)
        partial.replace(output)
        output_seconds = round(len(stitched) / sample_rate, 3)
        warnings = ["TTS ran on CPU; synthesis is slow on this host"] if tts.device == "cpu" else []
        staged = _Checkpoint(
            sample_rate=sample_rate,
            warnings=tuple(warnings),
            execution=ExecutionInfo(
                backend=str(tts.name), model_id=model.name, device=tts.device, model_revision=model.revision
            ),
            metrics={
                "generation_seconds": round(generation_seconds, 3),
                "processing_seconds": round(time.perf_counter() - started, 3),
                "output_seconds": output_seconds,
                "rtf": round(generation_seconds / output_seconds, 4) if output_seconds else None,
                "chunks": len(chunks),
                "failed_chunks": 0,
                "peak": round(peak(stitched), 4),
                "rms": round(rms(stitched), 5),
                "loudness_normalized": spec.options.loudness_normalization,
                "voice_version_id": str(spec.voice_version_id) if spec.voice_version_id else None,
            },
        )
        self._staged.commit(job, staged.to_metadata(), required=(output_name,))
        return staged

    async def _voice(self, spec: SynthesisSpec, work: Path) -> tuple[CloneReference | PresetSpeaker, InstalledModel]:
        """Return the voice conditioning and checkpoint, fetching a cloned voice's pinned reference to the workspace."""
        if spec.preset_speaker is not None:
            if not self._preset_checkpoint:
                raise InvalidRequestError("preset voices are not configured")
            return PresetSpeaker(name=spec.preset_speaker), self._model_host.tts_checkpoint(self._preset_checkpoint)
        if spec.voice_version_id is None:
            raise InvalidRequestError("cloned voice has no pinned version")
        async with self._uow() as uow:
            version = await uow.voices.get_version(spec.voice_version_id)
        if version is None or version.reference_uri is None or version.reference_backend is None:
            raise NotFoundError("the pinned voice version no longer exists")
        location = AssetLocation(StorageBackend(version.reference_backend), version.reference_uri)
        if location.backend is not self._store.backend:
            raise InvalidRequestError("the voice reference is stored in a different storage backend")
        suffix = Path(version.reference_uri).suffix or DEFAULT_REFERENCE_SUFFIX
        reference = await self._store.fetch(location, work / f"reference{suffix}")
        voice = CloneReference(voice_version_id=version.id, audio_path=reference, text=version.reference_text)
        return voice, self._model_host.tts_checkpoint(self._cloning_checkpoint)
