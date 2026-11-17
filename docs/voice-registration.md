# register_voice flow

A cloned voice has to be registered before `text_to_speech` can use it. Registration is a job like the others.
What makes it different is that the voice row is created **pending** at submission and only becomes **ready**
in the same transaction that completes the job.

## 1. Submission

```mermaid
sequenceDiagram
    autonumber
    participant C as MCP client
    participant T as register_voice tool
    participant JS as JobService
    participant PL as JobPlanner
    participant DB as PostgreSQL

    C->>T: register_voice {name, reference_audio_path, consent_confirmed, reference_text, language}
    T->>JS: submit(RegisterVoiceCommand, owner)
    JS->>PL: plan(command, owner)
    PL->>PL: require a TTS runtime, consent_confirmed must be true
    PL->>PL: validate name, InputPolicy.resolve_file(reference_audio_path)
    PL->>PL: conditioning mode: icl with reference_text, else xvector
    PL->>PL: new voice_id + voice_version_id, denoise if audio.denoising.enabled
    PL-->>JS: JobPlan(REGISTER_VOICE job, NewVoice with consent record)
    JS->>DB: INSERT voice (pending) + version + job (queued), COMMIT
    JS-->>T: Submission(job_id, voice_id)
    T-->>C: job_envelope + voice_id
```

## 2. Validation pipeline (`VoiceRegistrationService`)

```mermaid
flowchart TD
    Start([execute job]) --> Probe["DECODING 5%: ffprobe duration"]
    Probe --> Dur{"between MIN and MAX reference seconds?"}
    Dur -->|no| Bad1[[invalid_request: length]]
    Dur -->|yes| Wav["ffmpeg to 24 kHz mono reference.wav"]
    Wav --> Den{"spec.denoise?"}
    Den -->|yes| DFN["DENOISING 15%: 48 kHz, DeepFilterNet3, back to 24 kHz"]
    Den -->|no| Levels
    DFN --> Levels{"clipped above 1%, or RMS too low?"}
    Levels -->|yes| Bad2[[invalid_request: clipped or too quiet]]
    Levels -->|no| SttQ{"verification STT model installed?"}
    SttQ -->|no| Skip["warning: speech check skipped"]
    SttQ -->|yes| Check["VAD 25%: transcribe reference at 16 kHz"]
    Check --> Speech{"any speech?"}
    Speech -->|no| Bad3[[invalid_request: no speech]]
    Speech -->|yes| Sim{"reference_text given and similarity below 0.5?"}
    Sim -->|yes| Bad4[[invalid_request: transcript mismatch]]
    Sim -->|no| Cond
    Skip --> Cond["INFERENCING 45%: synthesize a validation sentence with this reference"]
    Cond --> Silent{"output is silence?"}
    Silent -->|yes| Bad5[[InferenceError]]
    Silent -->|no| Pub["PUBLISHING 85%: put reference.wav (voice_reference) + preserved original (input)"]
    Pub --> Done(["JobCompletion with voice_version_id"])
```

## 3. Completion makes the voice usable

```mermaid
sequenceDiagram
    participant JW as JobWorker
    participant LC as JobTransitions
    participant DB as PostgreSQL

    alt outcome
        JW->>LC: mark_completed(job, outcome)
        LC->>DB: BEGIN, job running to completed (guarded)
        LC->>DB: voices.mark_ready(version): pending to ready + reference fields
        alt voice no longer pending (deleted meanwhile)
            LC->>DB: ROLLBACK
            LC->>DB: job failed (not_found), voice untouched, no asset rows
            Note over DB: uploaded objects have no rows, so the orphan reconciler removes them
        else ready
            LC->>DB: INSERT both assets linked to the voice version (reference and original), COMMIT
        end
    else error (non-retryable, or attempts exhausted)
        JW->>LC: mark_failed(job, error)
        LC->>DB: job failed + pending voice to failed, one transaction
    end
```

| Event on the registration job | Voice |
|---|---|
| completed | pending to ready |
| failed (worker), interrupted with no attempts left (recovery), cancelled (CLI) | pending to failed |
| automatic retry, recoverable interruption | stays pending |
| `jobs retry` of a failed registration | failed to pending (refused if the voice was deleted) |

Each voice change happens in the same transaction as the job change and only if the job change applied.
`VoiceRepository.mark_ready` updates the voice row (pending to ready) before it attaches the reference fields. The
voice row lock therefore serializes completion with deletion and manual retry, and a deleted or failed voice doesn't
get its reference repopulated. `mark_deleted` refuses a pending voice, because its registration may still complete.
It can be repeated on a deleted voice, keeping the first `deleted_at`, so an interrupted cleanup can be retried. From then
on, `list_voices` shows the voice as ready, and `text_to_speech(voice_id=...)` pins its current version.
`python -m app voices delete VOICE_ID` purges both recordings of every version (see [cli.md](cli.md)).
