"""OAuth resource-server behavior of the streamable HTTP transport, using the real JwtTokenVerifier."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app.adapters.identity.oidc.verifier import JwtTokenVerifier
from app.core.settings import load_settings
from app.mcp_server.auth import CallerResolver
from app.mcp_server.server import build_mcp_server
from tests.harness import build_harness
from tests.mocks.clock import MockMonotonicClock
from tests.mocks.identity import MockJwksClient
from tests.unit.test_jwt_verifier import AUDIENCE, ISSUER, RSA_KEY, sign_token

BASE_URL = "http://127.0.0.1:8000"
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Return a test client for the HTTP app with OAuth enabled."""
    harness = build_harness(tmp_path)
    settings = load_settings(
        server={"transport": "streamable_http"},
        auth={"issuer_url": ISSUER, "audience": AUDIENCE, "resource_server_url": AUDIENCE},
        storage={"data_root": str(tmp_path / "data")},
    )
    verifier = JwtTokenVerifier(
        jwks_client=MockJwksClient({"rsa-1": RSA_KEY.public_key()}),
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=frozenset({"RS256"}),
        leeway_seconds=0,
        key_ttl_seconds=3600,
        min_refresh_seconds=30,
        clock=MockMonotonicClock(),
    )
    services = replace(harness.tool_services(), caller_resolver=CallerResolver(trusted_local=None))
    server = build_mcp_server(settings, services, verifier)
    app = server.streamable_http_app(host="127.0.0.1")
    with TestClient(app, base_url=BASE_URL) as test_client:
        yield test_client
    harness.executor.shutdown()


def test_missing_token_returns_401_with_resource_metadata(client: TestClient) -> None:
    """Answer 401 with WWW-Authenticate pointing at the protected resource metadata."""
    response = client.post("/mcp", json=INITIALIZE, headers=HEADERS)
    assert response.status_code == 401
    assert "resource_metadata=" in response.headers["www-authenticate"]


def test_protected_resource_metadata_lists_issuer(client: TestClient) -> None:
    """Publish RFC 9728 metadata naming the configured issuer."""
    response = client.get("/.well-known/oauth-protected-resource/mcp")
    assert response.status_code == 200
    body = response.json()
    assert body["authorization_servers"] == [ISSUER]
    assert body["resource"] == AUDIENCE


@pytest.mark.parametrize("claims", [{"aud": "http://other-api"}, {"iss": "https://evil"}, {"exp": 1}])
def test_invalid_tokens_return_401(client: TestClient, claims: dict[str, object]) -> None:
    """Reject wrong audience, wrong issuer and expired tokens with 401."""
    response = client.post(
        "/mcp", json=INITIALIZE, headers={**HEADERS, "Authorization": f"Bearer {sign_token(**claims)}"}
    )
    assert response.status_code == 401


def test_valid_token_without_scope_returns_403(client: TestClient) -> None:
    """Answer 403 insufficient_scope for a valid token without mcp:access."""
    token = sign_token(scope="openid profile")
    response = client.post("/mcp", json=INITIALIZE, headers={**HEADERS, "Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert "insufficient_scope" in response.headers.get("www-authenticate", "") + response.text


def test_valid_token_is_accepted(client: TestClient) -> None:
    """Accept a valid token with mcp:access."""
    response = client.post("/mcp", json=INITIALIZE, headers={**HEADERS, "Authorization": f"Bearer {sign_token()}"})
    assert response.status_code == 200
