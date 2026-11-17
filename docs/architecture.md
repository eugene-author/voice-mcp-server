# Architecture

The system is described in small diagrams instead of one large one. Each answers one question:

| # | Diagram | Question it answers |
|---|---|---|
| 1 | [Layers](#1-layers-and-dependency-direction) | Which package may depend on which? |
| 2 | [Request path](#2-request-path-mcp-call-to-queued-job) | What happens when an MCP tool is called? |
| 3 | [Job execution](#3-job-execution-on-the-worker) | How does a worker run a job? |
| 4 | [Inference and storage](#4-inference-and-storage) | Where do models run, and where do files go? |
| 5 | [Single process](#5-processes-and-background-tasks-role-all) | What runs inside one `serve` process? |
| 6 | [Split roles](#6-split-roles-api-and-gpu-worker) | How do the api and GPU worker cooperate, and how does on-demand start work? |

`app/main.py` is the composition root: it is the only place that knows the concrete adapters.

## 1. Layers and dependency direction

import-linter enforces these arrows. An arrow means "may import".

```mermaid
flowchart LR
    MCP["mcp_server<br/>tools, auth, errors"]
    Workers["workers<br/>worker, recovery,<br/>controller"]
    Services["services<br/>use cases"]
    Domain["domain<br/>pure types and rules"]
    Ports["ports<br/>Protocols"]
    Adapters["adapters<br/>vendors"]
    DB["db<br/>SQLAlchemy"]
    Main["main.py<br/>composition root"]

    MCP --> Services
    Workers --> Services
    Services --> Ports
    Services --> Domain
    Ports --> Domain
    Adapters --> Ports
    DB --> Ports
    Main -.-> MCP
    Main -.-> Workers
    Main -.-> Adapters
    Main -.-> DB
```

- Dotted arrows: `main.py` builds and wires every package; it is the only module allowed to import them all.
- `services`, `workers` and `mcp_server` don't import `adapters` or `db`; they only see `ports`.
- Adapter categories (`stt`, `tts`, `compute`, `storage`, …) don't import each other.

## 2. Request path: MCP call to queued job

```mermaid
flowchart TD
    Client["MCP client<br/>(Claude Code, agent, curl)"]
    SDK["MCPServer<br/>Streamable HTTP or stdio"]
    Verifier["JwtTokenVerifier<br/>(adapters/identity/oidc)"]
    IdP["Keycloak<br/>JWKS"]
    Tools["9 tools<br/>+ mapped_errors"]
    Caller[CallerResolver]
    JobService
    Queries[JobQueryService]
    Planner["JobPlanner<br/>InputPathPolicy<br/>TranscriptionOptionsResolver"]
    DB[("PostgreSQL")]
    Queue["JobQueue<br/>notify after commit"]

    Client -->|JSON-RPC + bearer token| SDK
    SDK --> Verifier
    Verifier -->|signing keys| IdP
    SDK --> Tools
    Tools --> Caller
    Tools -->|submit| JobService
    Tools -->|get_job, list_jobs| Queries
    JobService --> Planner
    JobService -->|commit queued job| DB
    JobService --> Queue
    Queries -->|read only| DB
```

- Submissions commit the job first, then wake the worker. A lost wake-up only delays the job: polling finds it.
- `get_job` and `list_jobs` only read PostgreSQL and don't touch models.

## 3. Job execution on the worker

```mermaid
flowchart TD
    Queue[JobQueue]
    Worker[JobWorker]
    Transitions["JobTransitions<br/>claim, complete, fail,<br/>retry, release"]
    Heartbeat
    DB[("PostgreSQL<br/>jobs, events")]
    STT[TranscriptionService]
    TTS[SynthesisService]
    Voice[VoiceRegistrationService]
    Staged["StagedJobAssets<br/>per-claim checkpoint"]
    Publisher[AssetPublisher]

    Queue -->|job id| Worker
    Worker -->|claim with new token| Transitions
    Worker --> Heartbeat
    Heartbeat -->|fenced by claim token| DB
    Transitions -->|fenced by claim token| DB
    Worker -->|speech_to_text| STT
    Worker -->|text_to_speech| TTS
    Worker -->|register_voice| Voice
    STT --> Staged
    TTS --> Staged
    STT --> Publisher
    TTS --> Publisher
    Voice --> Publisher
```

- Every write a worker makes to a running job requires its claim token, so a claim that lost its job changes nothing.
- Outputs are staged and committed before upload, so a retry republishes them without running inference again.

## 4. Inference and storage

```mermaid
flowchart LR
    Services["Transcription, Synthesis,<br/>VoiceRegistration services"]
    ModelHost["ModelHost<br/>load, evict, idle unload"]
    Runner["InferenceRunner<br/>single inference thread"]
    STT["SttBackend<br/>faster-whisper / mlx-whisper"]
    TTS["TtsEngine<br/>qwen-tts / mlx-audio"]
    DFN["DeepFilterNet<br/>(onnxruntime)"]
    VAD["Silero VAD<br/>(mlx path)"]
    FFmpeg[FfmpegMediaTool]
    Publisher[AssetPublisher]
    Store["AssetStore<br/>local or S3"]
    Catalog[HuggingFaceModelCatalog]
    Files[("DATA_ROOT<br/>models, workspace, assets")]
    S3[("S3 bucket")]

    Services --> ModelHost
    Services --> FFmpeg
    Services --> Publisher
    ModelHost --> Runner
    Runner --> STT
    Runner --> TTS
    Runner --> DFN
    Runner --> VAD
    ModelHost --> Catalog
    Catalog --> Files
    Publisher --> Store
    Store --> Files
    Store --> S3
```

- All native inference runs on one dedicated thread, so only one model works at a time.
- `models install` is the only step that downloads weights; at runtime models load from `DATA_ROOT/models`.

## 5. Processes and background tasks (role `all`)

`serve` (`app/main.py`) runs the MCP transport and `InProcessWorker.run()` in one `asyncio.TaskGroup`.

```mermaid
flowchart TD
    Serve["serve()"] --> Container["running_container()<br/>build adapters, services, workers"]
    Container -. "build fails" .-> Cleanup["AsyncExitStack<br/>unload models, stop executor,<br/>dispose engine"]
    Container --> TG{{asyncio.TaskGroup}}
    TG --> Transport["MCP transport<br/>HTTP or stdio"]
    TG --> Sup["InProcessWorker.run()"]
    Sup --> Lock{"worker lock<br/>acquired?"}
    Lock -->|no| Standby["standby<br/>reads and submissions only"]
    Lock -->|yes| Recover["startup recovery"]
    Recover --> Loops{{inner TaskGroup}}
    Loops --> WorkerLoop["JobWorker.run()"]
    Loops --> Poll["JobRecovery<br/>.poll_forever()"]
    Loops --> Orphans["AssetReconciler<br/>.run()"]
```

Only one process holds the PostgreSQL advisory lock and runs inference. Every other process serves tool calls (submissions, `get_job`, `list_jobs`), and the active worker picks those jobs up through recovery polling.

## 6. Split roles: api and GPU worker

`deployment.role` splits the same object graph across processes; see [gpu-worker.md](gpu-worker.md). The two processes share nothing but PostgreSQL and the asset store.

**6a. How a job crosses from the api to the worker**

```mermaid
flowchart LR
    subgraph API["api process"]
        Tools["MCP tools"] --> Announce["PostgresAnnouncing<br/>JobQueue"]
        Registry["RegistryRuntime<br/>Catalog"]
    end
    DB[("PostgreSQL<br/>jobs, workers")]
    subgraph GPU["worker process"]
        Listen["PostgresListening<br/>JobQueue"] --> Gpu["GpuWorker<br/>JobWorker"]
        Local["LocalRuntime<br/>Catalog"]
    end
    Announce -->|NOTIFY job id| DB
    DB -->|LISTEN| Listen
    Gpu -->|claims, heartbeats| DB
    Local -->|capability snapshot| DB
    DB -->|snapshot| Registry
```

- The api validates requests against the worker's last capability snapshot, so it needs no models.
- A missed `NOTIFY` only delays a job: the worker's `wake_queued` poll finds it.

**6b. How on-demand start and stop works**

```mermaid
flowchart LR
    Wake["WakeController<br/>(api, on-demand only)"]
    DB[("PostgreSQL<br/>jobs, workers")]
    Compute["ComputeTarget<br/>Docker or AWS ECS"]
    Gpu["GpuWorker<br/>(worker process)"]
    Recovery["JobRecovery<br/>AssetReconciler<br/>(api)"]

    Wake -->|1. reads demand<br/>and worker state| DB
    Wake -->|2. start, stop| Compute
    Compute -->|3. runs| Gpu
    Gpu -->|4. registers, drains,<br/>reports stopped| DB
    Wake -->|request or cancel drain| DB
    Recovery -->|stale leases,<br/>due retries| DB
```

- The controller is level-triggered: every tick it compares what should run with what runs, so new work during a shutdown simply leads to the next start.
- Recovery runs in the always-on api, so crashed jobs and due retries are handled while the GPU is stopped.

| Role | Built by | Background | Runtime catalog | Queue |
|---|---|---|---|---|
| `all` (default) | `build_container` | `InProcessWorker` as in §5, plus `IdleUnloader` when `idle_unload_seconds > 0` | `LocalRuntimeCatalog` | `AsyncioJobQueue` |
| `api` | `build_container` without `build_inference` | `PostgresAnnouncingJobQueue.run`, `RegistryRuntimeCatalog.run`, recovery polling, `AssetReconciler`, `WakeController.run` (only when `on_demand.enabled`) | `RegistryRuntimeCatalog` | `PostgresAnnouncingJobQueue` |
| `worker` | `gpu_worker_session` | `GpuWorker.run`: `JobWorker`, `wake_queued` poll, `PostgresListeningJobQueue.run`, `IdleUnloader`, registry heartbeat and drain | `LocalRuntimeCatalog` | `PostgresListeningJobQueue` |

The planner, options resolver, job, voice, model and health services depend on the `RuntimeCatalog` port (`app/ports/runtime.py`) instead of `ModelHost`, so the api validates requests from the worker's capability snapshot. Compute targets (`app/adapters/compute/`) are a separate adapter category, imported lazily and only when the master switch is on.