# CLI commands

Entry point: `python -m app` (Typer app in `app/cli.py`). Every command except `serve` loads settings with
transport `stdio`, so it doesn't need OAuth.

## Command map

```mermaid
flowchart LR
    CLI["python -m app"] --> serve
    CLI --> migrate
    CLI --> diagnose
    CLI --> models
    CLI --> jobs
    CLI --> voices
    CLI --> eval

    models --> mi["install --stt / --tts / --denoise [--backend] [--from DIR]"]
    models --> ml["list [--kind]"]
    jobs --> js["show JOB_ID"]
    jobs --> jr["result JOB_ID"]
    jobs --> jt["retry JOB_ID"]
    jobs --> jc["cancel JOB_ID"]
    voices --> vd["delete VOICE_ID"]
    eval --> ew["wer --dataset DIR [--model] [--language]"]

    serve -.-> PG[(PostgreSQL)]
    migrate -.-> PG
    diagnose -.-> PG
    js -.-> PG
    jr -.-> PG
    jt -.-> PG
    jc -.-> PG
    vd -.-> PG
    vd -.-> Store[("AssetStore")]
    mi -.-> HF["Hugging Face Hub or local --from dir"]
    mi -.-> Models[("DATA_ROOT/models")]
    ml -.-> Models
    diagnose -.-> Models
    ew -.-> Models
```

| Command | What it does | Touches |
|---|---|---|
| `serve [--transport] [--role all\|api]` | Runs the MCP server and background worker (`all`), or the MCP server without models (`api`; see [gpu-worker.md](gpu-worker.md)) | DB, models (all), store, IdP |
| `worker [--register-only] [--preload kind:name]` | Runs the GPU worker role until it drains (exit 0), or only publishes its capability snapshot | DB, models, store |
| `gpu status/start/drain/stop [--force]` | Inspects the GPU worker's compute and the controller's decision now, or acts once (stop refused while a job runs) | DB, compute target |
| `migrate [revision] [--create-database]` | `alembic upgrade` to `head` or a given revision; creates a missing database first when asked | DB |
| `diagnose [--skip-inference]` | Host profile, backend selection, installed models, DB check, a 1 s STT probe (`HealthCheckService.run_diagnostics`) | host, models, DB |
| `models install` | The only command that downloads weights | network or `--from`, models dir |
| `models list` | Registry entries and whether each is installed | models dir |
| `jobs show/result` | Prints a job without an ownership check (trusted local CLI) | DB |
| `jobs retry/cancel` | Manual transitions through `JobTransitions`, keeping a registration's voice in step | DB |
| `voices delete` | Marks the voice deleted, purges every stored recording of every version, then drops their rows | DB, store |
| `eval wer` | Offline WER/CER against `.txt` ground truth | models |

## serve

```mermaid
sequenceDiagram
    participant U as Operator
    participant CLI as cli.serve
    participant M as app.main
    participant DB as PostgreSQL
    participant IdP as OIDC provider

    U->>CLI: python -m app serve [--transport]
    CLI->>M: run(settings), configure logging, asyncio.run(serve)
    M->>M: build_inference: probe host, select STT/TTS backends, ModelHost
    M->>M: build_container: engine, UnitOfWork, AssetStore (S3 HeadBucket), queue, services, workers
    opt streamable_http
        M->>IdP: build_token_verifier, then warm() fetches JWKS
    end
    M->>M: build_mcp_server, register 9 tools
    par transport
        M->>M: run_streamable_http_async(host, port) or run_stdio_async()
    and background
        M->>DB: InProcessWorker: advisory lock, recovery, worker loop, reconciler
    end
```

## diagnose

```mermaid
flowchart TD
    Start([python -m app diagnose]) --> Session["diagnostics_session: offline runtime, data dirs, build_inference, DB engine if DATABASE_URL"]
    Session --> Run["HealthCheckService.run_diagnostics(include_inference)"]
    Run --> Facts["host facts, backend selection, installed models"]
    Facts --> DBCheck["database check: one query, or 'DATABASE_URL is not set'"]
    DBCheck --> Skip{"--skip-inference?"}
    Skip -->|yes| Release
    Skip -->|no| Probe["transcribe 1 s of silence with the smallest installed STT model"]
    Probe --> Release["session exit (also on error or Ctrl-C): unload models, stop inference thread, dispose engine"]
    Release --> Print([print JSON report])
```

The MCP `health_check` tool uses the same service's `check()`, which only runs dependency checks and doesn't load
weights or wait for the inference thread.

## models install

```mermaid
flowchart TD
    Start(["models install --stt/--tts/--denoise NAME"]) --> One{"exactly one kind given?"}
    One -->|no| Err[["exit 1: pass exactly one of ..."]]
    One -->|yes| Backend{"kind and --backend"}
    Backend -->|denoise| DF[deepfilternet]
    Backend -->|"explicit backend"| Given[use it]
    Backend -->|"auto, stt"| SelS["select_stt_backend(host): ctranslate2 or mlx"]
    Backend -->|"auto, tts"| SelT["select_tts_backend(host): torch or mlx"]
    DF --> Entry
    Given --> Entry
    SelS --> Entry
    SelT --> Entry
    Entry{"in models.yaml registry?"} -->|no| Err2[[invalid_request]]
    Entry -->|yes| Staging["mkdtemp staging dir next to target"]
    Staging --> From{"--from DIR?"}
    From -->|yes| Copy["copytree DIR (ignores .cache and .git)"]
    From -->|no| Download["snapshot_download(repo, pinned revision)"]
    Copy --> Manifest["manifest.json: sha256 per file, origin, revision, installed_at"]
    Download --> Manifest
    Manifest --> Swap["os.replace(staging, models/kind/backend/name)"]
    Swap --> Done([installed kind/backend/name])
```

The atomic `os.replace` means a half-finished download doesn't look installed: the runtime only sees a model
once its `manifest.json` is present.

## jobs retry / cancel

```mermaid
sequenceDiagram
    participant U as Operator
    participant CLI as cli
    participant LC as JobTransitions
    participant DB as PostgreSQL
    participant S as Running server

    U->>CLI: jobs retry JOB_ID (or cancel)
    CLI->>LC: retry(job_id) / cancel(job_id)
    LC->>DB: BEGIN
    opt retry of a register_voice job
        LC->>DB: voice still exists? (refuse if deleted)
    end
    LC->>DB: set_status(queued or cancelled), only if the state machine allows it
    alt transition applied
        LC->>DB: registration voice: failed to pending (retry) or pending to failed (cancel)
        LC->>DB: event "retried" / "cancelled", COMMIT
        CLI-->>U: job JOB_ID -> queued / cancelled
        S->>DB: JobRecovery poll finds the queued job (retry only), then claim and execute
    else not allowed (for example cancel while running, or the voice was deleted)
        LC->>DB: ROLLBACK
        CLI-->>U: exit 1
    end
```

## voices delete

```mermaid
sequenceDiagram
    participant U as Operator
    participant CLI as cli
    participant VS as VoiceService
    participant DB as PostgreSQL
    participant A as AssetStore

    U->>CLI: voices delete VOICE_ID
    CLI->>VS: delete_voice(voice_id, owner=None)
    VS->>DB: BEGIN, get voice (including already deleted)
    alt voice is pending (registration still running)
        VS-->>CLI: invalid_request: cancel or wait
    else ready, failed or deleted
        VS->>DB: mark deleted + clear version references
        VS->>DB: locations of every recording of every version (reference and original), COMMIT
        loop every stored recording
            VS->>A: delete(location) (missing object is fine)
        end
        VS->>DB: delete those asset rows, COMMIT
        VS-->>CLI: removed count
    end
    CLI-->>U: voice VOICE_ID deleted (N stored assets removed)
```

If a storage delete fails, the asset rows are still there and the voice is already hidden; running the command again
finishes the purge. Deleting an object that is already missing is not an error. Only the server-managed copies
(processed reference and preserved original) are purged; the caller's original input file isn't touched.

## eval wer

```mermaid
flowchart LR
    Start(["eval wer --dataset DIR"]) --> Build[build_inference, offline]
    Build --> Loop["for each audio file with a matching .txt"]
    Loop --> Conv[ffmpeg to 16 kHz WAV]
    Conv --> Tr[ModelHost.transcribe]
    Tr --> Score[WER and CER vs ground truth]
    Score --> Loop
    Score --> Out([JSON: per-file and overall WER/CER])
```
