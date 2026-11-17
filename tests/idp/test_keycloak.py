"""End-to-end token verification against the shipped Keycloak realm (docker compose --profile auth up keycloak)."""

import os

import httpx
import pytest

from app.adapters.identity.oidc.factory import build_token_verifier
from app.core.settings import AuthConfig

pytestmark = pytest.mark.idp
ISSUER = os.environ.get("KEYCLOAK_ISSUER", "http://127.0.0.1:8080/realms/voice")
AUDIENCE = "http://127.0.0.1:8000/mcp"
AGENT_SECRET = os.environ.get("KEYCLOAK_AGENT_SECRET", "")


async def _client_credentials_token() -> str:
    """Request an access token for the example headless agent."""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{ISSUER}/protocol/openid-connect/token",
            data={"grant_type": "client_credentials", "scope": "mcp:access"},
            auth=("voice-mcp-agent", AGENT_SECRET),
        )
    response.raise_for_status()
    return str(response.json()["access_token"])


@pytest.mark.skipif(not AGENT_SECRET, reason="set KEYCLOAK_AGENT_SECRET to the voice-mcp-agent client secret")
async def test_client_credentials_token_is_accepted() -> None:
    """Accept a Client Credentials token carrying mcp:access and the MCP audience."""
    verifier = await build_token_verifier(
        AuthConfig(issuer_url=ISSUER, audience=AUDIENCE, resource_server_url=AUDIENCE)
    )
    token = await verifier.verify_token(await _client_credentials_token())
    assert token is not None
    assert "mcp:access" in token.scopes
    assert token.resource == AUDIENCE


async def test_discovery_matches_issuer() -> None:
    """Discover the JWKS URI from the realm metadata."""
    verifier = await build_token_verifier(
        AuthConfig(issuer_url=ISSUER, audience=AUDIENCE, resource_server_url=AUDIENCE, warm_jwks_on_startup=True)
    )
    assert await verifier.verify_token("not-a-token") is None
