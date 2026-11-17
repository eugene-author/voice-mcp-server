#!/usr/bin/env bash
# Start the api with the on-demand worker in a local Docker container, for testing on Linux.
# Usage: start_on_demand.sh          production timers
#        FAST=1 start_on_demand.sh   30-second timers
# Ctrl+C stops the api and the worker. Production deployments should use Docker Compose and its socket proxy.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKER_CONTAINER="${WORKER_CONTAINER:-voice-local-worker}"
WORKER_IMAGE="${WORKER_IMAGE:-voice-mcp-local-worker:latest}"
DOCKER_ENDPOINT="unix:///var/run/docker.sock"
PORT="${VOICE_MCP_SERVER__PORT:-8000}"
UV_RUN=(uv run --no-sync python -m app)

log() { printf '%s\n' "$*" >&2; }
die() {
  printf 'FAIL  %s\n' "$*" >&2
  exit 1
}

# Read one value from the environment, then .env, then a default.
setting() {
  local name="$1" fallback="$2" value="${!1:-}"
  if [[ -z "$value" && -f .env ]]; then
    value=$(sed -n "s/^$name=//p" .env | tail -1 | tr -d "\"'\r")
  fi
  printf '%s' "${value:-$fallback}"
}

# Run a command silently, and show the end of its output if it fails.
quiet() {
  local output
  output=$(mktemp)
  if ! "$@" >"$output" 2>&1; then
    tail -3 "$output" >&2
    rm -f "$output"
    exit 1
  fi
  rm -f "$output"
}

# Print the directory holding the .venv's Python installation, which the worker container mounts.
host_python_root() {
  dirname "$(dirname "$(dirname "$(readlink -f .venv/bin/python)")")"
}

check_prerequisites() {
  [[ -x .venv/bin/python ]] || die "no .venv here; run uv sync first"
  case "$(host_python_root)" in
    / | /usr | /usr/local | /usr/local/*)
      die "the .venv uses a system Python; run: uv python install 3.14 && uv venv --python 3.14 --managed-python, then uv sync"
      ;;
  esac
  command -v docker >/dev/null || die "docker is not installed"
  command -v uv >/dev/null || die "uv is not installed"
  docker info >/dev/null 2>&1 || die "cannot talk to Docker; check that it is running and that you can use it"
}

check_port_is_free() {
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    die "port $PORT is in use; stop that server with Ctrl+C (if it was paused with Ctrl+Z, run fg first)"
  fi
}

check_keycloak() {
  curl -sf -o /dev/null --max-time 10 "$ISSUER_URL/.well-known/openid-configuration" ||
    die "cannot reach the identity provider at $ISSUER_URL; check that Keycloak is running"
}

build_worker_image() {
  docker image inspect "$WORKER_IMAGE" >/dev/null 2>&1 && return 0
  log "building $WORKER_IMAGE (first run only)"
  quiet docker build -t "$WORKER_IMAGE" - <<'DOCKERFILE'
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
DOCKERFILE
}

create_worker_container() {
  local root python_root
  if [[ "$(docker inspect -f '{{.State.Running}}' "$WORKER_CONTAINER" 2>/dev/null || true)" == true ]]; then
    die "$WORKER_CONTAINER is already running; stop it with: docker stop $WORKER_CONTAINER"
  fi
  docker rm "$WORKER_CONTAINER" >/dev/null 2>&1 || true
  root=$PWD
  python_root=$(host_python_root)
  docker create --name "$WORKER_CONTAINER" --network host --user "$(id -u):$(id -g)" -w "$root" --stop-timeout 60 \
    -v "$root:$root" -v "$python_root:$python_root:ro" \
    -e SERVICE_ROLE=worker -e HOME=/tmp -e GPU_ON_DEMAND=false \
    -e VOICE_MCP_JOBS__HEARTBEAT_SECONDS="${WORKER_HEARTBEAT_SECONDS}" \
    "$WORKER_IMAGE" "$root/.venv/bin/python" -m app worker >/dev/null
}

stop_worker() {
  log "stopping $WORKER_CONTAINER"
  docker stop -t 60 "$WORKER_CONTAINER" >/dev/null 2>&1 || true
}

ISSUER_URL=$(setting VOICE_MCP_AUTH__ISSUER_URL "http://127.0.0.1:8080/realms/voice")
RESOURCE_URL="http://127.0.0.1:$PORT/mcp"
if [[ "${FAST:-}" == 1 ]]; then
  IDLE_SECONDS=30 UPTIME_SECONDS=30 TICK_SECONDS=2 WORKER_HEARTBEAT_SECONDS=3
else
  IDLE_SECONDS=300 UPTIME_SECONDS=300 TICK_SECONDS=15 WORKER_HEARTBEAT_SECONDS=15
fi

check_prerequisites
check_port_is_free
check_keycloak

log "preparing the database"
quiet "${UV_RUN[@]}" migrate --create-database
log "preparing the worker container"
build_worker_image
create_worker_container
log "registering the worker"
quiet env GPU_ON_DEMAND=false "${UV_RUN[@]}" worker --register-only

export VOICE_MCP_AUTH__ISSUER_URL="$ISSUER_URL"
export VOICE_MCP_AUTH__AUDIENCE="$RESOURCE_URL"
export VOICE_MCP_AUTH__RESOURCE_SERVER_URL="$RESOURCE_URL"
export VOICE_MCP_SERVER__PORT="$PORT"
export GPU_ON_DEMAND=true
export VOICE_MCP_GPU_WORKER__ON_DEMAND__TARGET=docker
export VOICE_MCP_GPU_WORKER__ON_DEMAND__DOCKER__ENDPOINT="$DOCKER_ENDPOINT"
export VOICE_MCP_GPU_WORKER__ON_DEMAND__DOCKER__CONTAINER="$WORKER_CONTAINER"
export VOICE_MCP_GPU_WORKER__ON_DEMAND__TICK_SECONDS="$TICK_SECONDS"
export VOICE_MCP_GPU_WORKER__ON_DEMAND__IDLE_TIMEOUT_SECONDS="$IDLE_SECONDS"
export VOICE_MCP_GPU_WORKER__ON_DEMAND__MIN_UPTIME_SECONDS="$UPTIME_SECONDS"

log "starting the api on port $PORT; the worker starts when a job arrives. Press Ctrl+C to stop."
trap ':' INT TERM
"${UV_RUN[@]}" serve --role api --transport streamable_http || true
stop_worker
