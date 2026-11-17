"""Turns validated commands into frozen job rows (and a pending voice for registrations)."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.core.settings import Preset, Settings, select_preset
from app.domain.caller import Caller
from app.domain.errors import InvalidRequestError, NotFoundError
from app.domain.ids import JobId, VoiceId, new_job_id, new_voice_id, new_voice_version_id
from app.domain.jobs import (
    BatchTranscribeCommand,
    JobCommand,
    JobType,
    NewJob,
    RegisterVoiceCommand,
    SynthesisSpec,
    SynthesizeCommand,
    TranscribeCommand,
    VoiceRegistrationSpec,
)
from app.domain.speech_text import normalize_language
from app.domain.stt import Substitution
from app.domain.tts import TTS_LANGUAGES, ResolvedTtsOptions
from app.domain.voices import (
    PRESET_VOICE_PREFIX,
    ConsentRecord,
    NewVoice,
    VoiceStatus,
    conditioning_mode_for,
    ensure_usable,
    validate_voice_name,
)
from app.ports.clock import Clock
from app.ports.runtime import RuntimeCatalog
from app.ports.unit_of_work import UnitOfWorkFactory
from app.services.capabilities import HostCapabilities
from app.services.input_policy import InputPathPolicy
from app.services.runtime_selection import resolve_tts_options
from app.services.transcription_options_resolver import TranscriptionOptionsResolver

MAX_REQUEST_ID_LENGTH = 128


@dataclass(frozen=True, slots=True)
class JobPlan:
    """Rows to insert for one submission."""

    jobs: tuple[NewJob, ...]
    primary_job_id: JobId
    voice: NewVoice | None = None


class JobPlanner:
    """Validates commands against settings, presets, installed models and capabilities."""

    def __init__(
        self,
        *,
        settings: Settings,
        presets: Mapping[str, Preset],
        stt_options: TranscriptionOptionsResolver,
        runtime: RuntimeCatalog,
        input_policy: InputPathPolicy,
        uow: UnitOfWorkFactory,
        clock: Clock,
    ) -> None:
        """Store injected collaborators."""
        self._settings = settings
        self._presets = presets
        self._stt_options = stt_options
        self._runtime = runtime
        self._input_policy = input_policy
        self._uow = uow
        self._clock = clock

    async def plan(self, command: JobCommand, *, owner: Caller) -> JobPlan:
        """Return the job rows (and voice row) a command creates."""
        if command.request_id is not None and not 0 < len(command.request_id) <= MAX_REQUEST_ID_LENGTH:
            raise InvalidRequestError(f"request_id must be 1-{MAX_REQUEST_ID_LENGTH} characters")
        match command:
            case TranscribeCommand():
                return self._plan_transcription(command, owner)
            case BatchTranscribeCommand():
                return await self._plan_batch(command, owner)
            case SynthesizeCommand():
                return await self._plan_synthesis(command, owner)
            case RegisterVoiceCommand():
                return self._plan_registration(command, owner)

    def _snapshot(self, preset: str | None, **extra: Any) -> dict[str, Any]:
        """Return the effective configuration snapshot stored with a job, without secrets."""
        runtime = self._runtime.stt_runtime()
        tts = self._runtime.tts_backend()
        return {
            "preset": preset,
            "profile_stt_backend": str(runtime.backend),
            "stt_device": runtime.device,
            "stt_compute_type": runtime.compute_type,
            "tts_backend": str(tts) if tts else None,
            "storage_backend": self._settings.storage.backend,
            **extra,
        }

    def _new_job(
        self,
        job_type: JobType,
        owner: Caller,
        payload: dict[str, Any],
        snapshot: dict[str, Any],
        *,
        request_id: str | None,
        substitutions: tuple[Substitution, ...] = (),
        parent_job_id: JobId | None = None,
        job_id: JobId | None = None,
    ) -> NewJob:
        """Return a queued job row."""
        return NewJob(
            id=job_id or new_job_id(),
            type=job_type,
            owner=owner,
            input_payload=payload,
            config_snapshot=snapshot,
            created_at=self._clock.now(),
            max_attempts=self._settings.jobs.max_attempts,
            request_id=request_id,
            parent_job_id=parent_job_id,
            substitutions=substitutions,
        )

    def _plan_transcription(self, command: TranscribeCommand, owner: Caller) -> JobPlan:
        """Plan one speech_to_text job."""
        path = self._input_policy.resolve_file(command.input_path)
        resolved = self._stt_options.resolve(
            input_path=path,
            model=command.model,
            language=command.language,
            preset_name=command.preset,
            options=command.options,
        )
        job = self._new_job(
            JobType.SPEECH_TO_TEXT,
            owner,
            resolved.spec.to_payload(),
            self._snapshot(command.preset, model=resolved.spec.model),
            request_id=command.request_id,
            substitutions=resolved.substitutions,
        )
        return JobPlan(jobs=(job,), primary_job_id=job.id)

    async def _plan_batch(self, command: BatchTranscribeCommand, owner: Caller) -> JobPlan:
        """Plan a batch parent and one speech_to_text child per audio file; the directory walk runs in a thread."""
        directory = self._input_policy.resolve_directory(command.directory)
        files = await asyncio.to_thread(
            self._input_policy.audio_files,
            directory,
            recursive=command.recursive,
            limit=self._settings.jobs.batch_max_items,
        )
        if not files:
            raise InvalidRequestError("directory contains no supported audio files")
        parent_id = new_job_id()
        children: list[NewJob] = []
        substitutions: tuple[Substitution, ...] = ()
        for path in files:
            resolved = self._stt_options.resolve(
                input_path=path, model=None, language=None, preset_name=command.preset, options=None
            )
            substitutions = resolved.substitutions
            children.append(
                self._new_job(
                    JobType.SPEECH_TO_TEXT,
                    owner,
                    resolved.spec.to_payload(),
                    self._snapshot(command.preset, model=resolved.spec.model),
                    request_id=None,
                    substitutions=substitutions,
                    parent_job_id=parent_id,
                )
            )
        parent = self._new_job(
            JobType.TRANSCRIBE_BATCH,
            owner,
            {"directory": str(directory), "recursive": command.recursive, "files": [str(path) for path in files]},
            self._snapshot(command.preset),
            request_id=command.request_id,
            substitutions=substitutions,
            job_id=parent_id,
        )
        return JobPlan(jobs=(parent, *children), primary_job_id=parent_id)

    async def _plan_synthesis(self, command: SynthesizeCommand, owner: Caller) -> JobPlan:
        """Plan a text_to_speech job pinned to the current version of an owned or preset voice."""
        capabilities = HostCapabilities.from_runtime(self._runtime).require_tts("text_to_speech")
        text = command.text.strip()
        if not text:
            raise InvalidRequestError("text must not be empty")
        if len(text) > self._settings.tts.max_characters:
            raise InvalidRequestError(f"text exceeds {self._settings.tts.max_characters} characters")
        language = normalize_language(command.language)
        if language is not None and language not in TTS_LANGUAGES:
            raise InvalidRequestError(f"unsupported TTS language; supported: {', '.join(sorted(TTS_LANGUAGES))}")
        preset = select_preset(self._presets, command.preset)
        options = resolve_tts_options(
            requested=command.options,
            preset=preset.tts.to_options(),
            defaults=ResolvedTtsOptions(
                output_format=self._settings.tts.output_format,
                loudness_normalization=self._settings.audio.loudness_normalization,
                max_chunk_chars=self._settings.tts.max_chunk_chars,
                gap_ms=self._settings.tts.gap_ms,
            ),
        )
        if command.voice_id.startswith(PRESET_VOICE_PREFIX):
            if not capabilities.preset_voices or not self._settings.tts.preset_checkpoint:
                raise InvalidRequestError("preset voices are not available on this host")
            spec = SynthesisSpec(
                text=text,
                language=TTS_LANGUAGES[language or "auto"],
                voice_id=command.voice_id,
                voice_version_id=None,
                preset_speaker=command.voice_id.removeprefix(PRESET_VOICE_PREFIX),
                options=options,
            )
        else:
            async with self._uow() as uow:
                voice = await uow.voices.get(VoiceId(command.voice_id), owner)
            if voice is None or voice.status is not VoiceStatus.READY or voice.current_version is None:
                raise NotFoundError("voice not found or not ready")
            ensure_usable(voice.current_version, capabilities)
            spec = SynthesisSpec(
                text=text,
                language=TTS_LANGUAGES[language or voice.language or "auto"],
                voice_id=voice.id,
                voice_version_id=voice.current_version.id,
                preset_speaker=None,
                options=options,
            )
        job = self._new_job(
            JobType.TEXT_TO_SPEECH,
            owner,
            spec.to_payload(),
            self._snapshot(command.preset),
            request_id=command.request_id,
        )
        return JobPlan(jobs=(job,), primary_job_id=job.id)

    def _plan_registration(self, command: RegisterVoiceCommand, owner: Caller) -> JobPlan:
        """Plan a register_voice job and its pending voice record."""
        capabilities = HostCapabilities.from_runtime(self._runtime).require_tts("register_voice")
        if command.consent_confirmed is not True:
            raise InvalidRequestError(
                "consent_confirmed must be true: only register voices you are authorized to clone"
            )
        name = validate_voice_name(command.name)
        path = self._input_policy.resolve_file(command.reference_audio_path)
        reference_text = command.reference_text.strip() if command.reference_text else None
        mode = conditioning_mode_for(reference_text, capabilities)
        language = normalize_language(command.language)
        now = self._clock.now()
        voice_id, version_id = new_voice_id(), new_voice_version_id()
        spec = VoiceRegistrationSpec(
            voice_id=voice_id,
            voice_version_id=version_id,
            version=1,
            reference_audio_path=str(path),
            reference_text=reference_text,
            language=language,
            denoise=self._settings.audio.denoising.enabled and self._runtime.denoise_available(),
        )
        voice = NewVoice(
            voice_id=voice_id,
            voice_version_id=version_id,
            owner=owner,
            name=name,
            language=language,
            reference_text=reference_text,
            conditioning_mode=mode,
            consent=ConsentRecord(confirmed=True, confirmed_at=now, source="register_voice"),
            created_at=now,
        )
        job = self._new_job(
            JobType.REGISTER_VOICE,
            owner,
            spec.to_payload(),
            self._snapshot(None, conditioning_mode=str(mode)),
            request_id=command.request_id,
        )
        return JobPlan(jobs=(job,), primary_job_id=job.id, voice=voice)
