#!/usr/bin/env bash
# Check authorization and the MCP connection: tokens, 401s, session, the nine tools and health.
set -euo pipefail
source "$(dirname "$0")/lib/mcp.sh"
require_tools

EXPECTED_TOOLS="get_job health_check list_jobs list_models list_voices register_voice speech_to_text text_to_speech transcribe_batch"
PROBE='{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

probe_status() {
  curl -s --max-time 15 -D "$MCP_TMP/probe_headers" -o /dev/null -w '%{http_code}' "$MCP_URL" "$@" \
    -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -d "$PROBE" || true
}

status=$(probe_status)
[[ "$status" != 000 ]] || die "cannot reach the MCP server at $MCP_URL (run scripts/start_server.sh)"
[[ "$status" == 401 ]] || die "no token returned HTTP $status, expected 401"
grep -qi '^www-authenticate: Bearer' "$MCP_TMP/probe_headers" || die "the 401 has no WWW-Authenticate: Bearer header"
pass "no token -> 401 with WWW-Authenticate"

status=$(probe_status -H "Authorization: Bearer not-a-jwt")
[[ "$status" == 401 ]] || die "invalid token returned HTTP $status, expected 401"
pass "invalid token -> 401"

mcp_token
pass "Keycloak issued a token to $MCP_CLIENT_ID"

mcp_connect
pass "initialize opened an MCP session"

tools=$(mcp_call tools/list | jq -r '[.tools[].name] | sort | join(" ")')
[[ "$tools" == "$EXPECTED_TOOLS" ]] || die "unexpected tools: $tools"
pass "tools/list returned the 9 tools"

health=$(mcp_tool health_check) || die "health_check failed: $health"
overall=$(jq -r .status <<<"$health")
[[ "$overall" == ok || "$overall" == degraded ]] || die "health_check status is $overall: $health"
jq -e '.dependencies.database == "ok"' <<<"$health" >/dev/null || die "the database check failed: $health"
pass "health_check status=$overall worker=$(jq -r '.worker // "-"' <<<"$health") stt=$(jq -r '.stt.device // "-"' <<<"$health")"
