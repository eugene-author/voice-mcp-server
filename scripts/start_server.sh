#!/usr/bin/env bash
# Start PostgreSQL, Keycloak and the MCP server over HTTP for these tests; Ctrl+C stops the server.
# The variables apply to this process only, so a stdio setting in .env keeps working.
set -euo pipefail
cd "$(dirname "$0")/.."

POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-voice-mcp-postgres}"
KEYCLOAK_CONTAINER="${KEYCLOAK_CONTAINER:-voice-mcp-keycloak-1}"
ISSUER_URL="${ISSUER_URL:-http://127.0.0.1:8080/realms/voice}"
RESOURCE_URL="${RESOURCE_URL:-http://127.0.0.1:8000/mcp}"

docker start "$POSTGRES_CONTAINER" "$KEYCLOAK_CONTAINER" >/dev/null
until curl -sf -o /dev/null "$ISSUER_URL/.well-known/openid-configuration"; do
  sleep 1
done

export VOICE_MCP_SERVER__TRANSPORT=streamable_http
export VOICE_MCP_AUTH__MODE=oauth
export VOICE_MCP_AUTH__ISSUER_URL="$ISSUER_URL"
export VOICE_MCP_AUTH__AUDIENCE="$RESOURCE_URL"
export VOICE_MCP_AUTH__RESOURCE_SERVER_URL="$RESOURCE_URL"

.venv/bin/python -m app migrate --create-database
exec .venv/bin/python -m app serve --role all
