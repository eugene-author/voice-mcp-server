# text_to_speech flow

Text and a `voice_id` go in: either a cloned voice (`voice_...`) or a preset speaker (`preset:<name>`). Out comes
`speech.wav|mp3|flac|ogg`. Submission, worker claim, completion and inline waiting are the same as for
[speech_to_text](speech-to-text.md#2-worker-execution). This page covers what differs.

## 1. Tool call to job submission

```mermaid
sequenceDiagram
    autonumber
    participant C as MCP client
    participant T as text_to_speech tool
    participant JS as JobService
    participant CAP as HostCapabilities
    participant PL as JobPlanner
    participant DB as PostgreSQL
    participant Q as AsyncioJobQueue

    C->>T: tools/call text_to_speech {text, voice_id, language, preset, wait, options, request_id}
    T->>T: SynthesizeCommand, Caller from token or OS user
    T->>JS: submit(command, owner)
    JS->>CAP: require_for(command)
    CAP-->>JS: ok, or capability_unavailable (no TTS runtime)
    JS->>PL: plan(command, owner)
    PL->>PL: strip text, check tts.max_characters, language in TTS_LANGUAGES
    PL->>PL: resolve options (request, then preset, then settings: format, loudness, chunk size, gap)
    alt voice_id starts with "preset:"
        PL->>PL: require preset checkpoint configured, SynthesisSpec(preset_speaker)
    else cloned voice
        PL->>DB: voices.get(voice_id, owner)
        DB-->>PL: Voice with current_version
        PL->>PL: must be READY, ensure_usable on this runtime (MLX needs reference_text)
        PL->>PL: SynthesisSpec pinned to voice_version_id
    end
    PL-->>JS: JobPlan(one TEXT_TO_SPEECH job)
    JS->>DB: INSERT job (queued) + event, COMMIT
    JS->>Q: notify(job_id)
    JS-->>T: wait_for(...), then JobView
    T-->>C: job_envelope {job_id, status, files: {wav|mp3|...}}
```

Pinning `voice_version_id` at submission keeps a queued job on the exact reference it was planned with, even if
the voice is re-registered later.

## 2. The synthesis pipeline (`SynthesisService`)

```mermaid
flowchart TD
    Start([execute job]) --> Staged{"StagedJobAssets.load: marker, audio file and valid metadata?"}
    Staged -->|"yes: retry after publish failure"| Publish
    Staged -->|"no (incomplete or corrupt checkpoints are discarded)"| Voice{"preset_speaker?"}
    Voice -->|yes| Preset["PresetSpeaker + CustomVoice checkpoint"]
    Voice -->|no| Fetch["load pinned voice version from DB, AssetStore.fetch reference into the job workspace"]
    Fetch --> Clone["CloneReference(audio, reference_text) + Base checkpoint"]
    Preset --> Chunk
    Clone --> Chunk["chunk_text(text, max_chunk_chars)"]
    Chunk --> Loop["INFERENCING 10-80%: for each chunk, ModelHost.synthesize on the inference thread"]
    Loop --> Silent{"chunk RMS below silence threshold?"}
    Silent -->|yes| Retry[["InferenceError (retryable)"]]
    Silent -->|no| Rate{"same sample rate as earlier chunks?"}
    Rate -->|no| Fail[[InferenceError]]
    Rate -->|"yes, more chunks"| Loop
    Rate -->|"yes, last chunk"| Stitch["EXPORTING 82%: stitch chunks with gap_ms, stitched.wav (float)"]
    Stitch --> Loud{"loudness_normalization?"}
    Loud -->|yes| Norm["ffmpeg loudnorm, mastered.wav"]
    Loud -->|no| Encode
    Norm --> Encode["ffmpeg encode to output_format (.part, then rename)"]
    Encode --> Meta["commit checkpoint: outcome.json (metrics, execution), .complete last"]
    Meta --> Publish["PUBLISHING 92%: AssetPublisher puts speech.ext in AssetStore"]
    Publish --> Completion([JobCompletion: voice_id, duration_seconds, asset, metrics])
```

`ModelHost.synthesize` loads the checkpoint lazily: `Qwen3-TTS-12Hz-0.6B-Base` for cloned voices,
`...-CustomVoice` for presets. With exclusive residency it unloads the Whisper model first. The engine is
`qwen-tts` on torch (CUDA or CPU) or `mlx-audio` on Apple Silicon. Either way it only turns one text chunk into
samples. Chunking, stitching, mastering and encoding live in the service.

## 3. From MCP call to an audio file, end to end

```mermaid
sequenceDiagram
    autonumber
    participant C as MCP client
    participant T as text_to_speech tool
    participant JW as JobWorker
    participant S as SynthesisService
    participant R as ModelHost
    participant E as TtsEngine (Qwen3-TTS)
    participant F as FFmpeg
    participant A as AssetStore
    participant DB as PostgreSQL

    C->>T: text_to_speech(text, voice_id)
    T->>DB: commit queued job
    T-)JW: notify(job_id)
    JW->>DB: claim (running)
    JW->>S: execute(job, progress)
    S->>A: fetch voice reference (cloned voices)
    loop each text chunk
        S->>R: synthesize(checkpoint, SynthesisRequest)
        R->>E: load if needed, then synthesize(chunk)
        E-->>R: SynthesizedAudio(samples, sample_rate)
        R-->>S: audio
    end
    S->>F: normalize_loudness, then encode
    S->>A: put speech.wav (or mp3/flac/ogg)
    A-->>S: AssetLocation (file path or s3:// URI)
    S-->>JW: JobCompletion
    JW->>DB: assets + metrics + job completed, one transaction
    JW-->>T: waiter resolved
    T-->>C: {status: completed, files: {wav: ".../speech.wav"}}
```

If the job outlives the inline wait, the client gets `status: running` with `phase` and `progress`. It then calls
`get_job(job_id)` until `files` appears. In S3 mode, `include_download_urls=true` adds presigned URLs.
