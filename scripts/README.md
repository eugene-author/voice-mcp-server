# Test scripts

Bash scripts for Linux and macOS (Bash 3.2 or later) that test a running Voice MCP Server the way a real client does: OAuth token, MCP session over Streamable
HTTP, then tool calls. STT and TTS run as background jobs, and the scripts poll `get_job` every 3-5 seconds until the
job finishes.

## Setup

These need `curl`, `jq` and `ffprobe`, plus a server over HTTP with Keycloak. To start one locally:

```bash
scripts/start_server.sh
```

The scripts read `KEYCLOAK_ADMIN_PASSWORD` from the gitignored `.env` and use it to get the client secret from Keycloak.
An exported variable takes priority, and exporting `MCP_CLIENT_SECRET` skips Keycloak's admin API altogether.

## Tests

| Script | What it checks |
|---|---|
| `run_all.sh [--all]` | Runs 01 to 04 and stops on the first failure; `--all` adds 05 |
| `01_connect.sh` | 401 without or with a bad token, Keycloak token, MCP session, the nine tools, `health_check` |
| `02_add_voice.sh [sample]` | Registers a new voice from `<sample>.wav` and `<sample>.txt` in `INPUTS_DIR`, default `jessica`, and saves its ID |
| `03_stt.sh [file]` | Transcribes a file, by default `jessica.wav` in `INPUTS_DIR`, prints the transcript and checks the output files |
| `04_tts.sh ["text"]` | Synthesizes text with the saved voice, shows its name, and checks the audio with `ffprobe` |
| `05_batch.sh [directory]` | Transcribes a directory and checks every child job |

`start_on_demand.sh` starts the api with the worker in a local Docker container, which the api starts and stops by
itself. `FAST=1` shortens the idle timers to 30 seconds. It's for testing on Linux; press Ctrl+C to stop both.

A poll prints one line:

```text
[00:12] poll 3  status=running  phase=inferencing  progress=30%
```

During `inferencing` the percentage is an estimate, because the models report no progress of their own. It follows
how long similar work took recently and stays below the next real checkpoint.

Scripts exit non-zero on failure. Audio paths must be inside the server's allowed input directories.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `MCP_URL` | `http://127.0.0.1:8000/mcp` | MCP endpoint |
| `KEYCLOAK_URL` | `http://127.0.0.1:8080` | Keycloak base URL |
| `INPUTS_DIR` | `var/inputs/demo` | Where the default audio files are |
| `POLL_MIN_SECONDS`, `POLL_MAX_SECONDS` | `3`, `5` | Wait between polls |
| `POLL_TIMEOUT_SECONDS` | `900` | Stop polling after this long; the job keeps running |
| `STT_MODEL`, `STT_LANGUAGE`, `STT_PRESET`, `STT_DENOISE` | unset | Options for `03_stt.sh` |
| `TTS_VOICE_ID`, `TTS_LANGUAGE`, `TTS_FORMAT` | saved voice, `en`, server default | Options for `04_tts.sh` |
| `VOICE_REFERENCE`, `VOICE_TRANSCRIPT`, `VOICE_NAME`, `VOICE_CONSENT` | sample voice | Options for `02_add_voice.sh` |

Registering your own recording needs `VOICE_CONSENT=yes`, which confirms you are authorized to clone that voice.

The saved voice ID is in `var/test-scripts/state.env`. Voices belong to the OAuth client, so voices registered over
stdio don't appear here. Every `02_add_voice.sh` run adds another voice, also through `run_all.sh`. Test jobs and
voices stay in the database.
