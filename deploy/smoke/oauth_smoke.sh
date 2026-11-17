#!/usr/bin/env bash
# OAuth smoke test for a running Voice MCP Server in streamable_http mode with the reference Keycloak realm.
#
#   docker compose -f deploy/compose.yaml --profile auth up -d      # or a native server + Keycloak
#   KEYCLOAK_ADMIN_PASSWORD=... deploy/smoke/oauth_smoke.sh
#
# Checks: no token -> 401 with WWW-Authenticate, invalid token -> 401, a valid token for the MCP audience but
# without the mcp:access scope -> 403, and a valid agent token -> MCP initialize + health_check succeed.
# The 403 case uses a temporary client created through the Keycloak admin API and deleted on exit.
# Requires curl and jq. Exits non-zero on the first failed expectation.
set -euo pipefail

KEYCLOAK_URL="${KEYCLOAK_URL:-http://127.0.0.1:8080}"
REALM="${REALM:-voice}"
MCP_URL="${MCP_URL:-http://127.0.0.1:8000/mcp}"
KEYCLOAK_ADMIN="${KEYCLOAK_ADMIN:-admin}"
: "${KEYCLOAK_ADMIN_PASSWORD:?set KEYCLOAK_ADMIN_PASSWORD}"
TEMP_CLIENT="voice-mcp-smoke-noscope"
PROTOCOL_VERSION="2025-06-18"

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1" >&2; exit 1; }

admin_token=$(curl -sf -d client_id=admin-cli -d username="$KEYCLOAK_ADMIN" -d password="$KEYCLOAK_ADMIN_PASSWORD" \
  -d grant_type=password "$KEYCLOAK_URL/realms/master/protocol/openid-connect/token" | jq -r .access_token)
admin() { curl -sf -H "Authorization: Bearer $admin_token" -H "Content-Type: application/json" "$@"; }
client_uuid() { admin "$KEYCLOAK_URL/admin/realms/$REALM/clients?clientId=$1" | jq -r '.[0].id // empty'; }
client_secret() { admin "$KEYCLOAK_URL/admin/realms/$REALM/clients/$(client_uuid "$1")/client-secret" | jq -r .value; }
client_token() {
  curl -sf -u "$1:$2" -d grant_type=client_credentials \
    "$KEYCLOAK_URL/realms/$REALM/protocol/openid-connect/token" | jq -r .access_token
}

cleanup() {
  local id
  id=$(client_uuid "$TEMP_CLIENT" || true)
  if [[ -n "$id" ]]; then admin -X DELETE "$KEYCLOAK_URL/admin/realms/$REALM/clients/$id" || true; fi
}
trap cleanup EXIT

initialize_body='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"'"$PROTOCOL_VERSION"'","capabilities":{},"clientInfo":{"name":"oauth-smoke","version":"1.0"}}}'
post_status() {
  curl -s -o /dev/null -w '%{http_code}' "$MCP_URL" "$@" -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" -d "$initialize_body"
}

status=$(post_status)
[[ "$status" == 401 ]] || fail "no token returned $status, expected 401"
curl -s -D - -o /dev/null "$MCP_URL" -H "Content-Type: application/json" -d '{}' | grep -qi '^www-authenticate: Bearer' \
  || fail "401 without WWW-Authenticate header"
pass "missing token -> 401 with WWW-Authenticate"

status=$(post_status -H "Authorization: Bearer not-a-jwt")
[[ "$status" == 401 ]] || fail "invalid token returned $status, expected 401"
pass "invalid token -> 401"

cleanup
admin -X POST "$KEYCLOAK_URL/admin/realms/$REALM/clients" -d @- <<JSON
{"clientId": "$TEMP_CLIENT", "publicClient": false, "serviceAccountsEnabled": true, "standardFlowEnabled": false,
 "directAccessGrantsEnabled": false, "defaultClientScopes": [], "optionalClientScopes": [],
 "protocolMappers": [{"name": "audience", "protocol": "openid-connect", "protocolMapper": "oidc-audience-mapper",
   "config": {"included.custom.audience": "$MCP_URL", "access.token.claim": "true", "id.token.claim": "false"}}]}
JSON
noscope_token=$(client_token "$TEMP_CLIENT" "$(client_secret "$TEMP_CLIENT")")
status=$(post_status -H "Authorization: Bearer $noscope_token")
[[ "$status" == 403 ]] || fail "token without mcp:access returned $status, expected 403"
pass "valid token without mcp:access -> 403"

agent_token=$(client_token voice-mcp-agent "$(client_secret voice-mcp-agent)")
issuer=$(jq -Rr 'split(".")[1] | gsub("-";"+") | gsub("_";"/") | @base64d | fromjson | .iss' <<<"$agent_token")
headers=$(mktemp)
curl -s -D "$headers" -o /dev/null "$MCP_URL" -H "Authorization: Bearer $agent_token" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -d "$initialize_body"
session=$(awk 'tolower($1) == "mcp-session-id:" {print $2}' "$headers" | tr -d '\r')
rm -f "$headers"
[[ -n "$session" ]] || fail "initialize with a valid token did not return an Mcp-Session-Id"
pass "valid token (iss=$issuer) -> initialize, session $session"

call() {
  curl -s "$MCP_URL" -H "Authorization: Bearer $agent_token" -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" -H "Mcp-Session-Id: $session" \
    -H "MCP-Protocol-Version: $PROTOCOL_VERSION" -d "$1"
}
call '{"jsonrpc":"2.0","method":"notifications/initialized"}' >/dev/null
health=$(call '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"health_check","arguments":{}}}' \
  | sed -n 's/^data: //p' | jq '.result.structuredContent')
[[ $(jq -r .status <<<"$health") == ok ]] || fail "health_check status: $(jq -c . <<<"$health")"
idp=$(jq -r '.dependencies.identity_provider // .checks.identity_provider // "absent"' <<<"$health")
[[ "$idp" == ok ]] || fail "identity_provider check is '$idp': $(jq -c . <<<"$health")"
pass "health_check ok, identity_provider check ok"
