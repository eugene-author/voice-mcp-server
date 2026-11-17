# speech_to_text flow

A file path goes in. Out come `transcript.txt/.json/.srt/.vtt`, plus the text inline when the job finishes within
the inline wait.

## 1. Tool call to job submission

```mermaid
sequenceDiagram
    autonumber
    participant C as MCP client
    participant SDK as MCPServer (HTTP/stdio)
    participant V as JwtTokenVerifier
    participant T as speech_to_text tool
    participant P as CallerResolver
    participant JS as JobService
    participant PL as JobPlanner
    participant DB as PostgreSQL
    participant Q as AsyncioJobQueue
    participant W as AsyncioJobWaiters

    C->>SDK: tools/call speech_to_text {file_path, model, language, preset, wait, options, request_id}
    opt Streamable HTTP
        SDK->>V: verify_token(bearer)
        V-->>SDK: AccessToken (iss, sub, scopes) or 401/403
    end
    SDK->>T: invoke handler (wrapped by mapped_errors)
    T->>T: build TranscribeCommand
    T->>P: current()
    P-->>T: Caller(issuer, subject), or OS user on stdio
    T->>JS: submit(command, owner)
    JS->>JS: HostCapabilities.require_for(command)
    JS->>PL: plan(command, owner)
    PL->>PL: InputPolicy.resolve_file (realpath inside allowed dirs)
    PL->>PL: TranscriptionOptionsResolver: caller > preset > default, installed model, backend support
    PL-->>JS: JobPlan(one SPEECH_TO_TEXT job)
    JS->>DB: BEGIN, find_by_request_id (idempotency)
    alt request_id already used
        DB-->>JS: existing job_id
    else new request
        JS->>DB: INSERT job (queued) + event "submitted", COMMIT
        JS->>Q: notify(job_id), only after commit
    end
    JS-->>T: Submission(job_id)
    T->>JS: wait_for(job_id, timeout = wait ? inline_wait_seconds : 0)
    loop until terminal or timeout
        JS->>DB: JobQueryService.get(job_id, owner)
        JS->>W: wait(job_id, slice of at most 1 s)
    end
    JS-->>T: JobView
    T-->>SDK: job_envelope {job_id, status, progress/phase or text, files, warnings}
    SDK-->>C: CallToolResult (SSE "data:" line over HTTP)
```

Validation failures (`invalid_request`, `capability_unavailable`, `model_not_installed`, ...) are raised before
anything is committed. `mapped_errors` turns them into `isError: true` with `{"error": {"code", "message"}}`.

## 2. Worker execution

```mermaid
sequenceDiagram
    autonumber
    participant Q as AsyncioJobQueue
    participant JW as JobWorker
    participant LC as JobTransitions
    participant HB as Heartbeat
    participant TS as TranscriptionService
    participant DB as PostgreSQL
    participant W as AsyncioJobWaiters

    JW->>Q: get()
    Q-->>JW: job_id
    JW->>LC: claim(job_id, lease_owner)
    LC->>DB: atomic UPDATE queued to running, attempt+1, event "started"
    DB-->>LC: ClaimedJob, or None if already taken
    JW->>HB: keep_alive(job_id), background task
    JW->>TS: execute(job, progress) (the service is the registered handler)
    TS-->>JW: JobCompletion (or VoiceMcpError)
    JW->>HB: cancel
    alt outcome
        JW->>LC: mark_completed
        LC->>DB: INSERT assets, transcript, metrics, job completed, one transaction
    else retryable error with attempts left
        JW->>LC: schedule_retry, status retry_wait with next_attempt_at
    else error
        JW->>LC: mark_failed
    end
    JW->>W: resolve_if_terminal(job_id), wakes the waiting tool call
    JW->>JW: workspace.remove_job, aggregate parent batch if any
```

## 3. The transcription pipeline (`TranscriptionService`)

```mermaid
flowchart TD
    Start([execute job]) --> Staged{"StagedJobAssets.load: marker, all 4 files and valid metadata?"}
    Staged -->|"yes: retry after publish failure"| Reuse[Reuse staged transcript + outcome.json]
    Staged -->|"no (incomplete or corrupt checkpoints are discarded)"| Probe["DECODING 5%: ffprobe duration"]
    Probe --> TooLong{"duration > max_duration_seconds?"}
    TooLong -->|yes| MediaErr[[MediaError]]
    TooLong -->|no| Wav["ffmpeg to 16 kHz mono input.wav"]
    Wav --> Den{"options.denoise?"}
    Den -->|yes| DFN["DENOISING 12%: resample to 48 kHz, DeepFilterNet3 ONNX, back to 16 kHz"]
    Den -->|no| VadQ
    DFN --> VadQ{"backend has native VAD, or VAD disabled?"}
    VadQ -->|"yes (faster-whisper)"| Infer
    VadQ -->|"no (mlx-whisper)"| Silero["VAD 20%: Silero regions, SpeechTimeline, speech.wav"]
    Silero --> Empty{"timeline empty?"}
    Empty -->|yes| NoSpeech["empty result + warning no_speech_detected"]
    Empty -->|no| Infer
    Infer["INFERENCING 30%: ModelHost.transcribe on the inference thread"] --> Map{"standalone VAD used?"}
    Map -->|yes| Remap[map_to_source: segment times back to the original timeline]
    Map -->|no| Quality
    Remap --> Quality
    NoSpeech --> Quality
    Quality["EXPORTING 85%: segment quality flags, filler/repeat annotations, optional removals"] --> Export["write transcript.txt/.json/.srt/.vtt atomically, then commit: outcome.json, .complete last"]
    Export --> Publish
    Reuse --> Publish["PUBLISHING 92%: AssetPublisher puts files in AssetStore (local or S3)"]
    Publish --> Completion([JobCompletion: text preview, language, duration, assets, metrics])
```

`ModelHost.transcribe` loads the Whisper model lazily on first use and, on small hosts, evicts the TTS
checkpoint first (exclusive residency). On Linux the backend is faster-whisper (CTranslate2), on macOS mlx-whisper.

`transcribe_batch` rejects the whole submission, before any upload or job row, if an accepted file is empty or over
`audio.max_input_bytes`, naming that file. Otherwise it creates a parent job plus one `speech_to_text` child per file in the same transaction. Each
child runs through this same pipeline. After every terminal child, `BatchService.aggregate` updates the parent and
writes `manifest.json` once all children are done.
