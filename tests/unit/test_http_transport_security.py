"""Host and Origin validation of the Streamable HTTP transport, using the in-process HTTP app."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from app.adapters.identity.oidc.verifier import JwtTokenVerifier
from app.core.settings import Settings, load_settings
from app.mcp_server.auth import CallerResolver
from app.mcp_server.server import build_mcp_server, transport_security
from tests.harness import build_harness
from tests.mocks.clock import MockMonotonicClock
from tests.mocks.identity import MockJwksClient
from tests.unit.test_http_auth import HEADERS, INITIALIZE
from tests.unit.test_jwt_verifier import AUDIENCE, ISSUER, RSA_KEY, sign_token

AUTH = {"issuer_url": ISSUER, "audience": AUDIENCE, "resource_server_url": AUDIENCE}
type ClientFactory = Callable[..., TestClient]


def _settings(tmp_path: Path, **server: Any) -> Settings:
    """Return Streamable HTTP settings with the given server section values."""
    return load_settings(
        server={"transport": "streamable_http", **server}, auth=AUTH, storage={"data_root": str(tmp_path / "data")}
    )


@pytest.fixture
def make_client(tmp_path: Path) -> Iterator[ClientFactory]:
    """Return a factory for test clients of the HTTP app built from server settings, as serve() builds it."""
    harness = build_harness(tmp_path)
    clients: list[TestClient] = []

    def factory(**server: Any) -> TestClient:
        """Build the app for these server settings and return an entered test client."""
        settings = _settings(tmp_path, **server)
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
        app = build_mcp_server(settings, services, verifier).streamable_http_app(
            host=settings.server.host, transport_security=transport_security(settings.server)
        )
        client = TestClient(app, base_url="http://127.0.0.1:8000")
        clients.append(client.__enter__())
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)
    harness.executor.shutdown()


def _initialize(client: TestClient, **headers: str) -> int:
    """Send an authorized initialize request with extra headers and return the status code."""
    authorized = {**HEADERS, "Authorization": f"Bearer {sign_token()}", **headers}
    return client.post("/mcp", json=INITIALIZE, headers=authorized).status_code


def test_listed_host_is_accepted_and_unlisted_host_rejected(make_client: ClientFactory) -> None:
    """Accept a configured Host header and reject any other with 421 on a non-loopback bind."""
    client = make_client(host="0.0.0.0", allowed_hosts=["voice.example.com"])
    assert _initialize(client, Host="voice.example.com") == 200
    assert _initialize(client, Host="attacker.example") == 421
    assert _initialize(client, Host="127.0.0.1:8000") == 421


def test_unlisted_origin_is_rejected(make_client: ClientFactory) -> None:
    """Reject a browser Origin that is not configured with 403 and accept a configured one."""
    client = make_client(
        host="0.0.0.0", allowed_hosts=["voice.example.com"], allowed_origins=["https://voice.example.com"]
    )
    assert _initialize(client, Host="voice.example.com", Origin="https://voice.example.com") == 200
    assert _initialize(client, Host="voice.example.com", Origin="https://attacker.example") == 403


def test_defaults_allow_loopback_only(make_client: ClientFactory) -> None:
    """Accept loopback Host and Origin headers by default and reject a rebinding attacker's host."""
    client = make_client()
    assert _initialize(client, Host="127.0.0.1:8000", Origin="http://localhost:3000") == 200
    assert _initialize(client, Host="attacker.example") == 421
    assert _initialize(client, Host="127.0.0.1:8000", Origin="https://attacker.example") == 403


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "voice.internal"])
def test_non_loopback_bind_without_allowed_hosts_fails(tmp_path: Path, host: str) -> None:
    """Refuse settings that bind Streamable HTTP beyond loopback with no allowed hosts."""
    with pytest.raises(ValidationError, match="allowed_hosts"):
        _settings(tmp_path, host=host)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]"])
def test_loopback_bind_needs_no_allowed_hosts(tmp_path: Path, host: str) -> None:
    """Accept loopback binds with the loopback-only default."""
    assert _settings(tmp_path, host=host).server.binds_loopback


def test_stdio_ignores_the_http_bind(tmp_path: Path) -> None:
    """Allow a non-loopback host in settings when the transport is stdio, as admin commands in Docker load it."""
    settings = load_settings(server={"transport": "stdio", "host": "0.0.0.0"}, storage={"data_root": str(tmp_path)})
    assert not settings.server.binds_loopback
