#!/usr/bin/env bash
# Shared functions for the test scripts. Source this file; do not run it.
#

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MCP_URL="${MCP_URL:-http://127.0.0.1:8000/mcp}"
KEYCLOAK_URL="${KEYCLOAK_URL:-http://127.0.0.1:8080}"
KEYCLOAK_REALM="${KEYCLOAK_REALM:-voice}"
KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN:-admin}"
TOKEN_URL="${TOKEN_URL:-$KEYCLOAK_URL/realms/$KEYCLOAK_REALM/protocol/openid-connect/token}"
MCP_CLIENT_ID="${MCP_CLIENT_ID:-voice-mcp-agent}"
MCP_PROTOCOL_VERSION="${MCP_PROTOCOL_VERSION:-2025-06-18}"
MCP_CALL_TIMEOUT="${MCP_CALL_TIMEOUT:-60}"
INPUTS_DIR="${INPUTS_DIR:-$REPO_ROOT/var/inputs/demo}"
STATE_FILE="${STATE_FILE:-$REPO_ROOT/var/test-scripts/state.env}"
POLL_MIN_SECONDS="${POLL_MIN_SECONDS:-3}"
POLL_MAX_SECONDS="${POLL_MAX_SECONDS:-5}"
POLL_TIMEOUT_SECONDS="${POLL_TIMEOUT_SECONDS:-900}"
POLL_QUEUED_HINT_SECONDS="${POLL_QUEUED_HINT_SECONDS:-30}"
POLL_MAX_ERRORS="${POLL_MAX_ERRORS:-3}"

# Read one value from the gitignored .env; an exported variable takes priority.
env_file_value() {
  [[ -f "$REPO_ROOT/.env" ]] || return 0
  sed -n "s/^$1=//p" "$REPO_ROOT/.env" | tail -1 | tr -d "\"'\r"
}
KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD:-$(env_file_value KEYCLOAK_ADMIN_PASSWORD)}"

MCP_TMP="$(mktemp -d)"
chmod 700 "$MCP_TMP"

log() { printf '%s\n' "$*" >&2; }
pass() { printf 'PASS  %s\n' "$*" >&2; }
die() {
  printf 'FAIL  %s\n' "$*" >&2
  exit 1
}

require_tools() {
  local tool
  for tool in curl jq ffprobe "$@"; do
    command -v "$tool" >/dev/null || die "missing required tool: $tool"
  done
}

# Portable helpers: GNU tools first, then BSD or generic ones. MCP_FORCE_PORTABLE=1 skips the GNU branches for testing.
now_ms() {
  local value=""
  [[ "${MCP_FORCE_PORTABLE:-}" == 1 ]] || value=$(date +%s%3N)
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$value"
  elif command -v python3 >/dev/null; then
    python3 -c 'import time; print(int(time.time() * 1000))'
  else
    perl -MTime::HiRes=time -e 'printf "%d\n", time * 1000'
  fi
}

file_size() {
  if [[ "${MCP_FORCE_PORTABLE:-}" != 1 ]]; then
    stat -c %s "$1" 2>/dev/null && return
    stat -f %z "$1" 2>/dev/null && return
  fi
  wc -c <"$1" | tr -d ' '
}

abs_path() {
  if [[ "${MCP_FORCE_PORTABLE:-}" != 1 ]] && command -v realpath >/dev/null; then
    realpath "$1"
  else
    printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"
  fi
}

state_get() {
  [[ -f "$STATE_FILE" ]] || return 0
  sed -n "s/^$1=//p" "$STATE_FILE" | tail -1
}

state_set() {
  mkdir -p "$(dirname "$STATE_FILE")"
  touch "$STATE_FILE"
  local temporary
  temporary=$(mktemp "$STATE_FILE.XXXXXX")
  {
    grep -v "^$1=" "$STATE_FILE" || true
    printf '%s=%s\n' "$1" "$2"
  } >"$temporary"
  mv "$temporary" "$STATE_FILE"
}

urlencode() { jq -rn --arg value "$1" '$value | @uri'; }

resolve_client_secret() {
  if [[ -n "${MCP_CLIENT_SECRET:-}" ]]; then
    printf '%s' "$MCP_CLIENT_SECRET"
    return
  fi
  [[ -n "${KEYCLOAK_ADMIN_PASSWORD:-}" ]] ||
    die "add KEYCLOAK_ADMIN_PASSWORD to .env, or set MCP_CLIENT_SECRET"
  local form admin clients client_uuid
  form="client_id=admin-cli&grant_type=password&username=$(urlencode "$KEYCLOAK_ADMIN")"
  form+="&password=$(urlencode "$KEYCLOAK_ADMIN_PASSWORD")"
  admin=$(printf '%s' "$form" | curl -sf --max-time 15 -d @- \
    "$KEYCLOAK_URL/realms/master/protocol/openid-connect/token" | jq -r '.access_token // empty') ||
    die "cannot sign in to Keycloak at $KEYCLOAK_URL as $KEYCLOAK_ADMIN"
  [[ -n "$admin" ]] || die "cannot sign in to Keycloak at $KEYCLOAK_URL as $KEYCLOAK_ADMIN"
  printf 'Authorization: Bearer %s\n' "$admin" >"$MCP_TMP/admin_headers"
  clients=$(curl -sf --max-time 15 -H "@$MCP_TMP/admin_headers" \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/clients?clientId=$(urlencode "$MCP_CLIENT_ID")") ||
    die "cannot list Keycloak clients"
  client_uuid=$(jq -r '.[0].id // empty' <<<"$clients")
  [[ -n "$client_uuid" ]] || die "Keycloak has no client named $MCP_CLIENT_ID"
  curl -sf --max-time 15 -H "@$MCP_TMP/admin_headers" \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/clients/$client_uuid/client-secret" | jq -r '.value // empty'
  rm -f "$MCP_TMP/admin_headers"
}

mcp_token() {
  local now expiry secret form response code
  now=$(date +%s)
  expiry=$(cat "$MCP_TMP/token_expiry" 2>/dev/null || echo 0)
  ((now < expiry)) && return 0
  if [[ ! -f "$MCP_TMP/client_secret" ]]; then
    secret=$(resolve_client_secret)
    [[ -n "$secret" ]] || die "could not resolve the secret of $MCP_CLIENT_ID"
    printf '%s' "$secret" >"$MCP_TMP/client_secret"
  fi
  secret=$(cat "$MCP_TMP/client_secret")
  form="grant_type=client_credentials&client_id=$(urlencode "$MCP_CLIENT_ID")&client_secret=$(urlencode "$secret")"
  response="$MCP_TMP/token.json"
  code=$(printf '%s' "$form" | curl -s --max-time 15 -o "$response" -w '%{http_code}' -d @- "$TOKEN_URL") || code=000
  [[ "$code" == 000 ]] && die "cannot reach Keycloak at $TOKEN_URL (run scripts/start_server.sh)"
  [[ "$code" == 200 ]] || die "Keycloak refused the token request with HTTP $code; check the client secret"
  printf 'Authorization: Bearer %s\n' "$(jq -r .access_token "$response")" >"$MCP_TMP/auth_headers"
  printf '%s' "$((now + $(jq -r .expires_in "$response") - 30))" >"$MCP_TMP/token_expiry"
  rm -f "$response"
}

# Print the HTTP status of one POST; the body and response headers land in $MCP_TMP.
mcp_post() {
  local body="$1" code
  local -a headers=(-H "Content-Type: application/json" -H "Accept: application/json, text/event-stream"
    -H "MCP-Protocol-Version: $MCP_PROTOCOL_VERSION" -H "@$MCP_TMP/auth_headers")
  if [[ -f "$MCP_TMP/session" ]]; then
    headers+=(-H "Mcp-Session-Id: $(cat "$MCP_TMP/session")")
  fi
  code=$(curl -s --max-time "$MCP_CALL_TIMEOUT" -D "$MCP_TMP/response_headers" -o "$MCP_TMP/response_body" \
    -w '%{http_code}' "${headers[@]}" -d "$body" "$MCP_URL") || code=000
  printf '%s' "$code"
}

mcp_connect() {
  local body code session
  mcp_token
  rm -f "$MCP_TMP/session"
  body=$(jq -cn --arg version "$MCP_PROTOCOL_VERSION" '{jsonrpc: "2.0", id: 1, method: "initialize",
    params: {protocolVersion: $version, capabilities: {}, clientInfo: {name: "voice-test-scripts", version: "1.0"}}}')
  code=$(mcp_post "$body")
  [[ "$code" == 000 ]] && die "cannot reach the MCP server at $MCP_URL (run scripts/start_server.sh)"
  [[ "$code" == 200 ]] || die "MCP initialize returned HTTP $code"
  session=$(awk 'tolower($1) == "mcp-session-id:" {print $2}' "$MCP_TMP/response_headers" | tr -d '\r')
  [[ -n "$session" ]] || die "MCP initialize returned no Mcp-Session-Id"
  printf '%s' "$session" >"$MCP_TMP/session"
  code=$(mcp_post '{"jsonrpc":"2.0","method":"notifications/initialized"}')
  [[ "$code" == 202 ]] || die "notifications/initialized returned HTTP $code"
}

mcp_disconnect() {
  [[ -f "$MCP_TMP/session" && -f "$MCP_TMP/auth_headers" ]] || return 0
  curl -s --max-time 5 -o /dev/null -X DELETE -H "@$MCP_TMP/auth_headers" \
    -H "Mcp-Session-Id: $(cat "$MCP_TMP/session")" -H "MCP-Protocol-Version: $MCP_PROTOCOL_VERSION" "$MCP_URL" || true
}

cleanup() {
  mcp_disconnect
  rm -rf "$MCP_TMP"
}
trap cleanup EXIT

# Print the JSON-RPC result of one call; a 401 or 404 renews the token or session once and retries.
mcp_call() {
  local method="$1" params="${2:-}" body code attempt message
  [[ -n "$params" ]] || params='{}'
  body=$(jq -cn --arg method "$method" --argjson params "$params" --argjson id "$(now_ms)" \
    '{jsonrpc: "2.0", id: $id, method: $method, params: $params}')
  for attempt in 1 2; do
    mcp_token
    code=$(mcp_post "$body")
    case "$code" in
      200) break ;;
      401 | 404)
        ((attempt == 1)) || {
          log "$method returned HTTP $code after renewing the session"
          return 1
        }
        rm -f "$MCP_TMP/token_expiry"
        mcp_connect
        ;;
      000)
        log "cannot reach the MCP server at $MCP_URL"
        return 1
        ;;
      *)
        log "$method returned HTTP $code"
        return 1
        ;;
    esac
  done
  message=$(tr -d '\r' <"$MCP_TMP/response_body" | sed -n 's/^data: //p' | tail -1)
  [[ -n "$message" ]] || message=$(cat "$MCP_TMP/response_body")
  jq -e 'has("result")' <<<"$message" >/dev/null 2>&1 || {
    log "$method failed: $(jq -c '.error // .' <<<"$message" 2>/dev/null || echo "$message")"
    return 1
  }
  jq -c .result <<<"$message"
}

# Print a tool's structured result; it also prints the result of a rejected call, and then returns 1.
mcp_tool() {
  local name="$1" arguments="${2:-}" result
  [[ -n "$arguments" ]] || arguments='{}'
  result=$(mcp_call tools/call "$(jq -cn --arg name "$name" --argjson arguments "$arguments" \
    '{name: $name, arguments: $arguments}')") || return 1
  jq -c '.structuredContent // .' <<<"$result"
  [[ "$(jq -r '.isError // false' <<<"$result")" == false ]]
}

job_id_of() { jq -r '.job_id // empty' <<<"$1"; }

# Print a path relative to the repository root, as /var/assets/...
display_path() { printf '%s' "${1#"$REPO_ROOT"}"; }

clock() {
  local seconds="$1"
  printf '%02d:%02d' "$((seconds / 60))" "$((seconds % 60))"
}

# Sleep so that the next poll starts between POLL_MIN_SECONDS and POLL_MAX_SECONDS after this one started.
sleep_until_next_poll() {
  local poll_started_ms="$1" interval_ms remaining_ms
  interval_ms=$(((POLL_MIN_SECONDS * 1000) + RANDOM % ((POLL_MAX_SECONDS - POLL_MIN_SECONDS) * 1000 + 1)))
  remaining_ms=$((interval_ms - ($(now_ms) - poll_started_ms)))
  ((remaining_ms > 0)) || return 0
  sleep "$((remaining_ms / 1000)).$(printf '%03d' "$((remaining_ms % 1000))")"
}

# Poll get_job every POLL_MIN_SECONDS to POLL_MAX_SECONDS and print the final job on stdout.
# Returns 0 when completed, 1 when failed, cancelled or partly failed, and 2 on timeout.
mcp_poll_job() {
  local job_id="$1" start now poll_started_ms elapsed polls=0 errors=0 hinted=0 view status phase progress
  start=$(date +%s)
  log "polling job $job_id every ${POLL_MIN_SECONDS}-${POLL_MAX_SECONDS} s"
  while true; do
    poll_started_ms=$(now_ms)
    now=$((poll_started_ms / 1000))
    elapsed=$((now - start))
    polls=$((polls + 1))
    if ! view=$(mcp_tool get_job "$(jq -cn --arg id "$job_id" '{job_id: $id}')"); then
      errors=$((errors + 1))
      log "[$(clock "$elapsed")] poll $polls  get_job failed ($errors of $POLL_MAX_ERRORS): ${view:-no response}"
      ((errors < POLL_MAX_ERRORS)) || return 1
    else
      errors=0
      status=$(jq -r .status <<<"$view")
      phase=$(jq -r '.phase // "-"' <<<"$view")
      progress=$(jq -r 'if .status == "completed" then "100%" elif .progress == null then "-" else "\(.progress)%" end' <<<"$view")
      log "[$(clock "$elapsed")] poll $polls  status=$status  phase=$phase  progress=$progress"
      case "$status" in
        completed)
          printf '%s\n' "$view"
          return 0
          ;;
        failed | cancelled | completed_with_errors)
          log "job ended as $status: $(jq -c '.error // .counts // {}' <<<"$view")"
          printf '%s\n' "$view"
          return 1
          ;;
        queued)
          if ((hinted == 0 && elapsed >= POLL_QUEUED_HINT_SECONDS)); then
            hinted=1
            log "still queued after ${elapsed}s: check that a worker is active (a standby server runs no jobs)"
          fi
          ;;
      esac
    fi
    if ((elapsed >= POLL_TIMEOUT_SECONDS)); then
      log "timed out after ${elapsed}s; the job is still ${status:-unknown} on the server"
      return 2
    fi
    sleep_until_next_poll "$poll_started_ms"
  done
}

# Check every local path in the job's files and print them; other storage backends are skipped.
check_output_files() {
  local view="$1" path found=0
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    path="${path#file://}"
    if [[ "$path" != /* ]]; then
      log "skipped remote file $path"
      continue
    fi
    [[ -s "$path" ]] || die "output file is missing or empty: $path"
    found=$((found + 1))
    log "  $(display_path "$path") ($(file_size "$path") bytes)"
  done < <(jq -r '.files // {} | .. | strings' <<<"$view")
  ((found > 0)) || die "the job reported no output files"
  printf '%s' "$found"
}
