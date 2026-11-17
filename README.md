# Voice MCP Server (TTS, STT)

[![CI](https://github.com/eugene-author/voice-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/eugene-author/voice-mcp-server/actions/workflows/ci.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/protocol-MCP-0F766E)](https://modelcontextprotocol.io/)
[![Packaging with uv](https://img.shields.io/badge/packaging-uv-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![Release 0.1.0](https://img.shields.io/badge/release-0.1.0-2563EB)](https://github.com/eugene-author/voice-mcp-server/releases)
[![MIT License](https://img.shields.io/badge/license-MIT-16A34A)](LICENSE)

## 1. Overview

Give your AI agents a voice and a way to listen. Voice MCP Server turns recordings into timestamped transcripts and text into audio using Whisper and Qwen3-TTS on your own hardware. Choose a preset voice or clone one from a short reference recording.

Transcription and synthesis run as asynchronous jobs. Agents can submit work in the background, receive a job ID, and poll jobs for progress and results. PostgreSQL preserves job state across client disconnects and worker restarts.

Run everything in one process for local use, or keep the MCP server online on CPU or seperate on-demand GPU worker when jobs arrive.

![Voice MCP architecture with JWT validation, durable PostgreSQL jobs, and an on-demand GPU worker on Docker or AWS ECS](docs/diagrams/architecture.svg)

## 2. Features

- **Speech-to-text** produces timestamped transcripts and subtitles from individual files or batches.
- **Text-to-speech** generates audio with a built-in speaker or a voice cloned from an authorized reference recording.
- **Authentication** validates OIDC-issued bearer JWTs and the `mcp:access` scope on Streamable HTTP. Jobs and voices are scoped to the caller. Local `stdio` calls run under the OS user.
- **Durable jobs** support background submission, status and progress polling, retries, and recovery of interrupted work.
- **GPU on demand** starts a separate worker for queued jobs or scheduled operating periods, then drains and stops it when idle. Running the worker for 10% of low-demand time cuts its runtime resources by 90%.
- **Local or S3 storage** holds transcripts, audio outputs, and voice references.
- **Extensible adapter** architecture provides interfaces for integrating additional speech models, inference backends, storage providers, and GPU worker targets.

### 2.1. Platform profiles

| Profile | Speech-to-text | Text-to-speech |
|---|---|---|
| `linux-cpu` | faster-whisper on CPU with INT8 | Qwen3-TTS with PyTorch on CPU |
| `linux-cuda` | faster-whisper on NVIDIA CUDA | Qwen3-TTS with PyTorch on CUDA |
| `macos-native` | mlx-whisper on Apple Silicon | Qwen3-TTS via mlx-audio |

The server selects the backend automatically. Use `health_check` or `python -m app diagnose` to see the selected profile, devices, and installed models.

### 2.2. Text-to-speech (TTS)

Call `text_to_speech` with text and a `voice_id` from `list_voices`. Preset speakers use the Qwen3-TTS CustomVoice checkpoint. Cloned voices use the Base checkpoint and a reference prepared by `register_voice`, which can apply DeepFilterNet denoising. Cloning on MLX requires the matching reference transcript.

`SynthesisService` splits long text into chunks, synthesizes and stitches them, then uses FFmpeg to master and encode the result as WAV, MP3, FLAC, or OGG. The completed job returns the audio location. [TTS pipeline details](docs/text-to-speech.md)

![TTS pipeline from the MCP call and voice selection through SynthesisService to a published audio file](docs/diagrams/tts-pipeline.svg)

### 2.3. Speech-to-text (STT)

Call `speech_to_text` with an allowed audio file. `TranscriptionService` decodes it with FFmpeg, optionally denoises it with DeepFilterNet, detects speech with Silero VAD when enabled, and runs Whisper. Results include a text preview and TXT, JSON, SRT, and VTT files. JSON includes segment timing and word timestamps when enabled.

Choose a preset (`fast`, `balanced`, `accurate`, or `studio`) or set options such as `denoise` per request. `transcribe_batch` creates a transcription job for each audio file and publishes a manifest when the batch finishes. [STT pipeline details](docs/speech-to-text.md)

![STT pipeline from an audio file through TranscriptionService, optional denoising, and Whisper to transcripts and subtitles](docs/diagrams/stt-pipeline.svg)

### 2.4. GPU on demand

API accepts requests while the GPU worker is stopped. It validates jobs against the worker's last published capabilities and stores them in PostgreSQL. PostgreSQL `LISTEN/NOTIFY` signals a running worker, and recovery polls pick up missed job.

`WakeController` starts the worker when runnable jobs arrive or a scheduled warm window begins. The default idle timeout is five minutes. During drain, the worker stops claiming jobs and finishes the current one. If the drain deadline expires, the job goes back to the queue.

One inference worker is active per database. The controller starts and stops a Docker container or sets the ECS worker service's desired count to zero or one. A managed ECS capacity provider can also scale in the underlying GPU EC2 instances when the Auto Scaling group allows a minimum of zero. [Worker lifecycle](docs/gpu-worker.md) · [ECS capacity providers](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/asg-capacity-providers.html)

## 3. Prerequisites

- **Native setup** uses Python 3.14, [uv](https://docs.astral.sh/uv/), FFmpeg with `ffprobe`, and PostgreSQL 16 or newer.
- **Docker deployment** uses Docker Engine and Compose v2. NVIDIA deployments also need a compatible host driver and the NVIDIA Container Toolkit.
- **Model installation** needs access to Hugging Face or a local copy of the weights and disk space for models and audio assets.
- **Streamable HTTP** requires an OIDC provider. Run the included Keycloak realm with Compose for the local HTTP demo.
- **Demo scripts** require Bash, `curl`, `jq`, and `ffprobe`.

## 4. Project layout

```text
.
├── app/
│   ├── adapters/          # Models, FFmpeg, identity, storage, and compute
│   ├── config/            # Defaults, model registry, and presets
│   ├── db/                # PostgreSQL models and repositories
│   ├── domain/            # Job, audio, and voice types and rules
│   ├── mcp_server/        # MCP tools, transport, and caller identity
│   ├── migrations/        # Alembic migrations
│   ├── ports/             # Interfaces for infrastructure adapters
│   ├── services/          # Transcription, synthesis, voices, and job submission
│   ├── workers/           # Job execution, recovery, and GPU wake control
│   ├── cli.py             # Commands for setup, serving, diagnostics, and worker control
│   └── main.py            # Connects application components and runs the MCP server or worker
├── deploy/                # Docker Compose, Keycloak, and AWS guidance
├── docs/                  # Architecture and pipeline documentation
├── scripts/               # MCP client smoke tests
├── tests/                 # Unit, contract, integration, and real-model tests
├── pyproject.toml         # Package metadata and tooling
└── uv.lock                # Locked dependencies
```

## 5. Installation and configuration

### 5.1. Clone and configure

```bash
git clone https://github.com/eugene-author/voice-mcp-server.git
cd voice-mcp-server
cp .env.example .env
```

For a native install, you need a PostgreSQL login that matches `DATABASE_URL`. This command starts a server that matches the template:

```bash
docker run -d --name voice-postgres -p 127.0.0.1:5434:5432 \
  -e POSTGRES_USER=voice -e POSTGRES_PASSWORD=change-me -e POSTGRES_DB=voice postgres:16.6
```

If you use your own server, put its login in `DATABASE_URL`. `python -m app migrate --create-database` creates the database if it is missing and the login is allowed to; plain `migrate` only applies migrations and reports a missing database.

For local Keycloak, set `KEYCLOAK_ADMIN_PASSWORD` in `.env`. For the Compose PostgreSQL service, set `POSTGRES_PASSWORD`.

| Setting | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection for jobs, voices, and worker state. |
| `DATA_ROOT` | Runtime data directory. Defaults to `./var` for native runs. |
| `VOICE_MCP_SERVER__TRANSPORT` | `stdio` for local clients or `streamable_http` for authenticated HTTP. |
| `VOICE_MCP_SERVER__ALLOWED_INPUT_DIRECTORIES` | JSON list of absolute input directories. Defaults to `DATA_ROOT/inputs`. |
| `VOICE_MCP_AUTH__ISSUER_URL` | Issuer URL exactly as it appears in JWTs. |
| `VOICE_MCP_AUTH__AUDIENCE`, `VOICE_MCP_AUTH__RESOURCE_SERVER_URL` | Both set to the MCP endpoint URL that clients use. |
| `STORAGE_BACKEND` | `local` by default, or `s3` with bucket and region configured. |

See the [environment template](.env.example) and [configuration defaults](app/config/default.yaml) for the full settings.

### 5.2. Install dependencies and models

Install Python with `uv python install 3.14`, then select the dependency extras for your host.

| Host | Command |
|---|---|
| Linux CPU | `uv sync --frozen --extra tts --extra cpu --extra denoise` |
| Linux NVIDIA (CUDA) | `uv sync --frozen --extra tts --extra cuda --extra denoise` |
| Apple Silicon | `uv sync --frozen --extra mlx --extra denoise` |

Run migrations, install the default Whisper model and the Qwen3-TTS cloning checkpoint, and check the setup.

```bash
uv run --no-sync python -m app migrate --create-database
uv run --no-sync python -m app models install --stt small
uv run --no-sync python -m app models install --tts Qwen3-TTS-12Hz-0.6B-Base
uv run --no-sync python -m app diagnose
```

Install `--tts Qwen3-TTS-12Hz-0.6B-CustomVoice` for preset speakers and `--denoise deepfilternet3` for optional noise reduction using the same `models install` command. Use `--from /path/to/weights` to install from local weights. Inference uses the installed weights offline.

### 5.3. Start the MCP server

Configure a local MCP client to launch this command from the repository root for `stdio`.

```bash
uv run --no-sync python -m app serve --transport stdio
```

For authenticated HTTP and the demo, start the reference Keycloak realm, then run the native server over Streamable HTTP.

```bash
docker compose --env-file .env -f deploy/compose.yaml --profile auth up -d --wait keycloak
uv run --no-sync python -m app serve --transport streamable_http
```

Connect at `http://127.0.0.1:8000/mcp`. The realm includes `voice-mcp-interactive` for interactive clients with PKCE and `voice-mcp-agent` for clients using the client credentials flow. Use an access token with audience `http://127.0.0.1:8000/mcp` and scope `mcp:access`. For a remote HTTP deployment, configure the public MCP URL and the matching Host and, for browser clients, Origin allowlists. [Reference Keycloak realm](deploy/keycloak/realm-voice.json)

### 5.4. Separate GPU worker on demand

The API and the worker can run as separate processes that share one PostgreSQL database. The API accepts jobs and the worker runs them. You can start the worker manually, or let the API start and stop it on demand.

#### Start the worker manually

Stop any server running with the default `all` role first. It holds port 8000 and the single worker lock.

On the worker host, after installing weights, register the worker once. This lets the API accept jobs while the worker is stopped. Repeat it whenever the installed models change.

```bash
uv run --no-sync python -m app worker --register-only
```

Then start each process in its own terminal:

```bash
uv run --no-sync python -m app serve --role api
uv run --no-sync python -m app worker
```

Jobs submitted while the worker is stopped stay `queued`, and it runs them when it starts.

#### Start and stop it on demand

The [startup script](scripts/start_on_demand.sh) runs the MCP API on the host and manages a worker container through the local Docker socket. 

From the repository root, start PostgreSQL and Keycloak, wait for the `voice` realm, and launch the API.

```bash
docker start voice-postgres voice-mcp-keycloak-1

until curl -fs -o /dev/null --max-time 5 \
  http://127.0.0.1:8080/realms/voice/.well-known/openid-configuration; do
  sleep 2
done

bash scripts/start_on_demand.sh
```

The script applies database migrations, builds the Python 3.14 worker image with FFmpeg if needed, and creates a stopped container. The worker uses host networking and the invoking user's UID and GID. The script mounts the checkout and host Python installation at their original paths, so the worker can use the host's `.venv` and models. The script needs a `.venv` built with a uv-managed or pyenv Python, because it mounts that installation into the worker container; it stops with instructions if `.venv` uses a system Python.

The script runs `worker --register-only` on the host to publish the installed model capabilities, then starts the Streamable HTTP API in the foreground at `http://127.0.0.1:8000/mcp` by default. The controller starts the container for queued jobs or scheduled operating periods. Outside scheduled periods, it requests a drain once the idle timeout and minimum uptime have both elapsed.

For a shorter test, run `FAST=1 bash scripts/start_on_demand.sh`.

#### Docker on one NVIDIA host

Set absolute `INPUTS_DIR` and `ASSETS_DIR` paths in `.env` and set `DOCKER_GID` to the Docker socket's group ID. Give the container user (UID `10001`) read access to inputs and write access to the model volume for installation, workspace volumes, and the assets directory. [Base Compose file](deploy/compose.yaml) · [CUDA overlay](deploy/compose.gpu.yaml)

Use the Compose `admin` service to install weights into the shared model volume. Register the worker before starting the API with on-demand control enabled.

```bash
compose_gpu() {
  docker compose --env-file .env \
    -f deploy/compose.yaml -f deploy/compose.gpu.yaml \
    --profile auth --profile split "$@"
}

compose_gpu up -d --wait postgres keycloak
compose_gpu run --rm migrate
compose_gpu --profile tools run --rm -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  admin models install --stt small --backend ctranslate2
compose_gpu --profile tools run --rm -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  admin models install --tts Qwen3-TTS-12Hz-0.6B-Base --backend torch
compose_gpu create gpu-worker
compose_gpu run --rm --no-deps gpu-worker worker --register-only
GPU_ON_DEMAND=true compose_gpu up -d api docker-proxy
```

The socket proxy gives the controller access to inspect, start, and stop the worker container. [Compose setup and operator commands](docs/gpu-worker.md#running-it-with-docker-compose)

#### AWS ECS across hosts

Run the API on CPU and the worker as a separate ECS service on GPU EC2. Set `STORAGE_BACKEND=s3` with the same bucket and region on both services. Install the `aws` and `s3` extras on the API and grant its task role `ecs:DescribeServices` and `ecs:UpdateService` on the worker service. The API ingests input audio into S3 so the two hosts can share assets.

```dotenv
SERVICE_ROLE=api
GPU_ON_DEMAND=true
VOICE_MCP_STORAGE__INPUT_TRANSFER=ingest
VOICE_MCP_GPU_WORKER__ON_DEMAND__TARGET=aws_ecs
VOICE_MCP_GPU_WORKER__ON_DEMAND__AWS_ECS__REGION=eu-west-1
VOICE_MCP_GPU_WORKER__ON_DEMAND__AWS_ECS__CLUSTER=voice
VOICE_MCP_GPU_WORKER__ON_DEMAND__AWS_ECS__SERVICE=voice-gpu-worker
```

Use `health_check` or `python -m app gpu status` to see the worker state and controller decisions. [AWS configuration and IAM example](deploy/aws/README.md#8-api-on-cpu-gpu-worker-on-demand)

## 6. MCP tool reference

Use `tools/list` to discover the full argument schemas. The table shows the inputs you'll use most often.

| Tool | Main arguments | Result |
|---|---|---|
| `health_check` | None | Server, database, storage, runtime, and worker status. |
| `list_models` | `kind` (optional) | Models by backend, with install and runtime status. |
| `list_voices` | `language` (optional) | Available preset speakers and the caller's cloned voices. |
| `register_voice` | `name`, `reference_audio_path`, `consent_confirmed`, optional `reference_text` | Registration job and `voice_id`. Use the voice after it becomes `ready`. |
| `speech_to_text` | `file_path`, optional `model`, `language`, `preset`, `options` | Transcription job with a text preview and transcript files on completion. |
| `text_to_speech` | `text`, `voice_id`, optional `language`, `preset`, `options` | Synthesis job with the rendered audio location on completion. |
| `transcribe_batch` | `directory`, optional `recursive`, `preset` | Parent job with child progress and a final `manifest.json`. |
| `get_job` | `job_id`, optional `include_download_urls` | Job status, progress, and output locations. Presigned URLs require S3 and presigning enabled. |
| `list_jobs` | Optional `status`, `type`, `parent_job_id`, `limit`, `cursor` | Newest jobs first, scoped to the caller, with cursor pagination. |

Use `wait: false` on job submissions to return immediately, then poll `get_job` with the returned ID. Use a stable `request_id` when retrying the same submission to avoid creating a duplicate job. Input arguments are absolute paths visible to the server.

## 7. Example requests and usage

### 7.1. Run the demo

With the HTTP server and Keycloak running, open a second terminal at the repository root. With the default data root, copy the included recording and transcript to a demo subdirectory.

```bash
demo_inputs="$PWD/var/inputs/demo"
mkdir -p "$demo_inputs"
cp tests/fixtures/demo/jessica.wav tests/fixtures/demo/jessica.txt "$demo_inputs/"
chmod +x scripts/0[1-4]_*.sh
bash scripts/run_all.sh
```

![Demo of run_all.sh running authentication, voice registration, transcription, and synthesis](https://github.com/user-attachments/assets/8154d366-0cf1-457f-b97c-70f0810ea199)

| Script | What it shows |
|---|---|
| `01_connect.sh` | Check token enforcement, open an MCP session, list tools, and check health. |
| `02_add_voice.sh` | Register the reference voice and save its ID when ready. |
| `03_stt.sh` | Transcribe the recording and verify the output files. |
| `04_tts.sh` | Synthesize a new sentence with the saved voice and verify the audio. |

The scripts read the local Keycloak admin password from `.env`. Set `MCP_CLIENT_SECRET` to use an existing client secret. [Demo script options](scripts/README.md)

### 7.2. Call the tools directly

In an initialized MCP session, pass these objects as `params` to `tools/call`. Replace the example paths with absolute paths inside an allowed input directory.

```json
{
  "name": "speech_to_text",
  "arguments": {
    "file_path": "/absolute/path/to/voice-mcp-server/var/inputs/demo/jessica.wav",
    "preset": "balanced",
    "wait": false
  }
}
```

To synthesize speech with a cloned voice, first register a reference recording you have permission to use. This example uses the demo recording and its exact transcript.

```json
{
  "name": "register_voice",
  "arguments": {
    "name": "Jessica",
    "reference_audio_path": "/absolute/path/to/voice-mcp-server/var/inputs/demo/jessica.wav",
    "reference_text": "Hello, it's nice to meet you. I would like to tell you a great story today.",
    "language": "en",
    "consent_confirmed": true,
    "wait": false
  }
}
```

When the registration job reaches `completed`, use the `voice_id` returned by `register_voice` in the synthesis request.

```json
{
  "name": "text_to_speech",
  "arguments": {
    "text": "The voice worker is ready.",
    "voice_id": "<voice-id>",
    "options": {"output_format": "wav"},
    "wait": false
  }
}
```

Poll `get_job` with the returned `job_id` for any submission. Completed transcription and synthesis jobs include output locations in `files`.

```json
{"name": "get_job", "arguments": {"job_id": "<returned-job-id>"}}
```

## 8. Tests

```bash
uv run --no-sync pytest
TEST_DATABASE_URL='postgresql://voice:password@127.0.0.1:5433/voice_test' \
  uv run --no-sync pytest -m "integration or s3"
```

Use a disposable PostgreSQL database for integration tests because they truncate application tables. S3 tests use a simulated bucket through Moto. The real-model suites need the corresponding dependency extras, FFmpeg, and installed weights.

| Suite | Command |
|---|---|
| Linux CPU models | `uv run --no-sync pytest -m models -v` |
| Apple Silicon models | `uv run --no-sync pytest -m "mlx or models" -v` |
| NVIDIA CUDA models | `uv run --no-sync pytest -m gpu -v` |
## 9. License

The server code is released under the [MIT License](LICENSE). Dependencies and model weights are distributed under their respective upstream licenses.

## 10. Disclaimer

The software is provided as is. Use voice cloning only with the speaker's consent and process recordings you have permission to use. Review generated audio and transcripts before publishing or relying on them.
