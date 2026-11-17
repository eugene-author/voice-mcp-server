"""Voice registration jobs: validate, check the transcript and publish; the voice stays pending until done."""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from app.domain.assets import AssetRole, UploadedInput, job_input_key, voice_reference_key
from app.domain.audio import clipped_fraction, rms
from app.domain.errors import InferenceError, InvalidRequestError, ModelNotInstalledError
from app.domain.jobs import ClaimedJob, ExecutionInfo, JobCompletion, JobPhase, ProgressCallback, VoiceRegistrationSpec
from app.domain.speech_text import transcript_similarity
from app.domain.stt import SttRequest, VadSettings
from app.domain.tts import TTS_LANGUAGES, CloneReference, SynthesisRequest
from app.domain.voices import MAX_REFERENCE_SECONDS, MIN_REFERENCE_SECONDS
from app.ports.media import MediaConverter, MediaProbe
from app.services.inference_progress import InferenceProgress
from app.services.input_transfer import fetch_input
from app.services.job_workspace import JobWorkspace
from app.services.model_host import ModelHost
from app.services.publishing import AssetPublisher, StagedFile

logger = logging.getLogger(__name__)
REFERENCE_SAMPLE_RATE = 24000
STT_SAMPLE_RATE = 16000
MAX_CLIPPED_FRACTION = 0.01
MIN_REFERENCE_RMS = 0.003
MIN_TEXT_SIMILARITY = 0.5
VALIDATION_TEXT = "This is a short check of the new voice."
REFERENCE_WAV_FILE = "reference.wav"
REFERENCE_48K_WAV_FILE = "reference_48k.wav"
REFERENCE_DENOISED_48K_WAV_FILE = "reference_denoised_48k.wav"
REFERENCE_DENOISED_WAV_FILE = "reference_denoised.wav"
REFERENCE_16K_WAV_FILE = "reference_16k.wav"


class VoiceRegistrationService:
    """Executes register_voice jobs; voice records stay runtime-portable (audio, transcript, metadata)."""

    def __init__(
        self,
        *,
        probe: MediaProbe,
        converter: MediaConverter,
        model_host: ModelHost,
        publisher: AssetPublisher,
        workspace: JobWorkspace,
        cloning_checkpoint: str,
        verification_stt_model: str,
        inference_progress: InferenceProgress,
    ) -> None:
        """Store injected collaborators."""
        self._probe = probe
        self._converter = converter
        self._model_host = model_host
        self._publisher = publisher
        self._workspace = workspace
        self._cloning_checkpoint = cloning_checkpoint
        self._inference_progress = inference_progress
        self._verification_stt_model = verification_stt_model

    async def execute(self, job: ClaimedJob, progress: ProgressCallback) -> JobCompletion:
        """Validate and publish the reference, and mark the pinned version ready on completion."""
        spec = VoiceRegistrationSpec.from_payload(job.input_payload)
        work = self._workspace.claim_dir(job)
        uploaded = UploadedInput.from_payload(job.input_payload)
        source = await fetch_input(self._publisher, uploaded, spec.reference_audio_path, work)
        await progress(JobPhase.DECODING, 5)
        info = await self._probe.probe(source)
        if not MIN_REFERENCE_SECONDS <= info.duration_seconds <= MAX_REFERENCE_SECONDS:
            raise InvalidRequestError(
                f"reference must be {MIN_REFERENCE_SECONDS:.0f}-{MAX_REFERENCE_SECONDS:.0f} seconds long "
                f"(got {info.duration_seconds:.1f} s); 5-15 s of clean speech works best"
            )
        reference = await self._converter.to_wav(source, work / REFERENCE_WAV_FILE, sample_rate=REFERENCE_SAMPLE_RATE)
        if spec.denoise:
            await progress(JobPhase.DENOISING, 15)
            reference = await self._denoise(reference, work)
        self._check_levels(reference)
        await progress(JobPhase.VAD, 25)
        warnings, similarity = await self._check_speech(spec, reference, work)
        await progress(JobPhase.INFERENCING, 45)
        checkpoint = self._model_host.tts_checkpoint(self._cloning_checkpoint)
        language = TTS_LANGUAGES.get(spec.language or "auto", "auto")
        voice = CloneReference(voice_version_id=spec.voice_version_id, audio_path=reference, text=spec.reference_text)
        async with self._inference_progress.track(
            progress,
            phase=JobPhase.INFERENCING,
            start=45,
            end=85,
            units=len(VALIDATION_TEXT),
            key=checkpoint.name,
            learn=self._model_host.loaded_models["tts"] == checkpoint.name,
        ):
            audio = await self._model_host.synthesize(
                checkpoint, SynthesisRequest(text=VALIDATION_TEXT, language=language, voice=voice)
            )
        if rms(audio.samples) < MIN_REFERENCE_RMS / 3:
            raise InferenceError("validation synthesis with this reference produced silence")
        await progress(JobPhase.PUBLISHING, 85)
        files = [
            StagedFile(
                name="reference",
                path=reference,
                role=AssetRole.VOICE_REFERENCE,
                key=voice_reference_key(job.owner, spec.voice_id, spec.version, "wav", claim_token=job.claim_token),
                duration_seconds=round(info.duration_seconds, 3),
                sample_rate=REFERENCE_SAMPLE_RATE,
            ),
        ]
        if uploaded is None:
            extension = source.suffix.lstrip(".") or "bin"
            key = job_input_key(job.owner, job.id, extension, claim_token=job.claim_token)
            files.append(StagedFile(name="original", path=source, role=AssetRole.INPUT, key=key))
        assets = await self._publisher.publish(files)
        if uploaded is not None:
            assets = (*assets, uploaded.as_original())
        tts = self._model_host.tts
        return JobCompletion(
            result={
                "voice_id": spec.voice_id,
                "conditioning_mode": "icl" if spec.reference_text else "xvector",
                "reference_seconds": round(info.duration_seconds, 3),
                "text_similarity": similarity,
            },
            assets=assets,
            warnings=tuple(warnings),
            execution=ExecutionInfo(
                backend=str(tts.name) if tts else "",
                model_id=checkpoint.name,
                device=tts.device if tts else "",
                model_revision=checkpoint.revision,
            ),
            voice_version_id=spec.voice_version_id,
        )

    async def _denoise(self, reference: Path, work: Path) -> Path:
        """Denoise the reference at DeepFilterNet's rate and return it at the reference rate."""
        upsampled = await self._converter.to_wav(reference, work / REFERENCE_48K_WAV_FILE, sample_rate=48000)
        await self._model_host.denoise(upsampled, work / REFERENCE_DENOISED_48K_WAV_FILE)
        return await self._converter.to_wav(
            work / REFERENCE_DENOISED_48K_WAV_FILE,
            work / REFERENCE_DENOISED_WAV_FILE,
            sample_rate=REFERENCE_SAMPLE_RATE,
        )

    @staticmethod
    def _check_levels(reference: Path) -> None:
        """Reject clipped or near-silent references."""
        samples, _ = sf.read(str(reference), dtype="float32")
        samples = np.asarray(samples, dtype=np.float32)
        if clipped_fraction(samples) > MAX_CLIPPED_FRACTION:
            raise InvalidRequestError("the reference recording is clipped; record at a lower level")
        if rms(samples) < MIN_REFERENCE_RMS:
            raise InvalidRequestError("the reference recording is too quiet")

    async def _check_speech(
        self, spec: VoiceRegistrationSpec, reference: Path, work: Path
    ) -> tuple[list[str], float | None]:
        """Transcribe the reference with VAD to confirm speech and that it matches reference_text."""
        try:
            model = self._model_host.stt_model(self._verification_stt_model)
        except ModelNotInstalledError:
            return ["speech and transcript check skipped: STT model is not installed"], None
        speech = await self._converter.to_wav(reference, work / REFERENCE_16K_WAV_FILE, sample_rate=STT_SAMPLE_RATE)
        capabilities = self._model_host.stt.capabilities
        request = SttRequest(
            audio_path=speech,
            language=spec.language,
            beam_size=5 if capabilities.beam_search else 1,
            temperature=0.0,
            word_timestamps=False,
            vad=replace(VadSettings(), enabled=capabilities.native_vad),
        )
        result = await self._model_host.transcribe(model, request)
        if not result.text.strip():
            raise InvalidRequestError("no speech was detected in the reference recording")
        if not spec.reference_text:
            return [], None
        similarity = round(transcript_similarity(spec.reference_text, result.text), 3)
        if similarity < MIN_TEXT_SIMILARITY:
            raise InvalidRequestError(
                f"reference_text does not match the recording (similarity {similarity:.2f}); "
                "provide the exact words spoken"
            )
        return [], similarity
