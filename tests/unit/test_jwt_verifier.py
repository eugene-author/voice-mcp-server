"""JwtTokenVerifier against locally generated keys, a MockJwksClient and a MockMonotonicClock."""

import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from app.adapters.identity.oidc import factory
from app.adapters.identity.oidc.factory import key_endpoint
from app.adapters.identity.oidc.verifier import JwtTokenVerifier
from app.core.settings import AuthConfig
from app.domain.errors import ConfigurationError
from tests.mocks.clock import MockMonotonicClock
from tests.mocks.identity import MockJwksClient

ISSUER = "https://idp.example/realms/voice"
AUDIENCE = "http://127.0.0.1:8000/mcp"
RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
EC_KEY = ec.generate_private_key(ec.SECP256R1())


def sign_token(*, kid: str = "rsa-1", key: Any = RSA_KEY, algorithm: str = "RS256", **claims: Any) -> str:
    """Return a signed token with sensible default claims."""
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-1",
        "exp": now + 300,
        "scope": "openid mcp:access",
        "azp": "cli",
    }
    payload.update(claims)
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, key, algorithm=algorithm, headers={"kid": kid})


@pytest.fixture
def mock_jwks_client() -> MockJwksClient:
    """Return a key set with one RSA and one EC key."""
    return MockJwksClient({"rsa-1": RSA_KEY.public_key(), "ec-1": EC_KEY.public_key()})


def build_verifier(
    *,
    jwks_client: MockJwksClient,
    clock: MockMonotonicClock,
    min_refresh_seconds: int = 30,
    key_ttl_seconds: int = 3600,
) -> JwtTokenVerifier:
    """Return a verifier for the test issuer and audience."""
    return JwtTokenVerifier(
        jwks_client=jwks_client,
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=frozenset({"RS256", "ES256"}),
        leeway_seconds=0,
        key_ttl_seconds=key_ttl_seconds,
        min_refresh_seconds=min_refresh_seconds,
        clock=clock,
    )


async def test_valid_rsa_token_maps_claims(
    mock_jwks_client: MockJwksClient, mock_monotonic: MockMonotonicClock
) -> None:
    """Accept a valid RS256 token and map scopes, client, subject and resource."""
    token = await build_verifier(jwks_client=mock_jwks_client, clock=mock_monotonic).verify_token(sign_token())
    assert token is not None
    assert (token.subject, token.client_id, token.resource) == ("user-1", "cli", AUDIENCE)
    assert token.scopes == ["openid", "mcp:access"]
    assert token.claims == {"iss": ISSUER}


async def test_valid_ec_token_and_scp_claim(
    mock_jwks_client: MockJwksClient, mock_monotonic: MockMonotonicClock
) -> None:
    """Accept ES256 and read scopes from an scp list."""
    raw = sign_token(kid="ec-1", key=EC_KEY, algorithm="ES256", scope=None, scp=["mcp:access"])
    token = await build_verifier(jwks_client=mock_jwks_client, clock=mock_monotonic).verify_token(raw)
    assert token is not None
    assert token.scopes == ["mcp:access"]


@pytest.mark.parametrize(
    "claims",
    [
        {"exp": int(time.time()) - 10},
        {"nbf": int(time.time()) + 600},
        {"aud": "http://other-api"},
        {"iss": "https://evil.example"},
        {"sub": None},
    ],
    ids=["expired", "not-yet-valid", "wrong-audience", "wrong-issuer", "missing-sub"],
)
async def test_invalid_claims_rejected(
    claims: dict[str, Any], mock_jwks_client: MockJwksClient, mock_monotonic: MockMonotonicClock
) -> None:
    """Reject tokens with invalid standard claims."""
    verifier = build_verifier(jwks_client=mock_jwks_client, clock=mock_monotonic)
    assert await verifier.verify_token(sign_token(**claims)) is None


async def test_alg_none_and_hmac_rejected(mock_jwks_client: MockJwksClient, mock_monotonic: MockMonotonicClock) -> None:
    """Reject unsigned and HMAC-signed tokens before any key lookup."""
    verifier = build_verifier(jwks_client=mock_jwks_client, clock=mock_monotonic)
    unsigned = jwt.encode({"iss": ISSUER, "sub": "x"}, None, algorithm="none", headers={"kid": "rsa-1"})
    hmac = jwt.encode(
        {"iss": ISSUER, "sub": "x"}, "secret-key-that-is-long-enough-32b", algorithm="HS256", headers={"kid": "rsa-1"}
    )
    assert await verifier.verify_token(unsigned) is None
    assert await verifier.verify_token(hmac) is None
    assert await verifier.verify_token("not.a.jwt") is None
    assert mock_jwks_client.fetch_count == 0


async def test_unknown_kid_inside_refresh_interval_is_rejected_without_fetch(
    mock_jwks_client: MockJwksClient, mock_monotonic: MockMonotonicClock
) -> None:
    """Reject a second unknown key ID within the interval without another key-set fetch."""
    verifier = build_verifier(jwks_client=mock_jwks_client, clock=mock_monotonic, min_refresh_seconds=30)
    await verifier.verify_token(sign_token(kid="unknown-1"))
    mock_monotonic.advance(seconds=5)

    result = await verifier.verify_token(sign_token(kid="unknown-2"))

    assert result is None
    assert mock_jwks_client.fetch_count == 1


async def test_rotated_key_validates_after_interval(
    mock_jwks_client: MockJwksClient, mock_monotonic: MockMonotonicClock
) -> None:
    """Start accepting a newly published key once the refresh interval passed."""
    verifier = build_verifier(jwks_client=mock_jwks_client, clock=mock_monotonic, min_refresh_seconds=30)
    new_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = sign_token(kid="rsa-2", key=new_key)
    assert await verifier.verify_token(token) is None
    mock_jwks_client.keys["rsa-2"] = new_key.public_key()
    mock_monotonic.advance(seconds=31)
    assert await verifier.verify_token(token) is not None


async def test_cached_key_avoids_fetch_and_expires_after_ttl(
    mock_jwks_client: MockJwksClient, mock_monotonic: MockMonotonicClock
) -> None:
    """Serve known keys from cache and stop accepting removed keys after the TTL."""
    verifier = build_verifier(jwks_client=mock_jwks_client, clock=mock_monotonic, key_ttl_seconds=60)
    await verifier.verify_token(sign_token())
    await verifier.verify_token(sign_token())
    assert mock_jwks_client.fetch_count == 1
    del mock_jwks_client.keys["rsa-1"]
    mock_monotonic.advance(seconds=61)
    assert await verifier.verify_token(sign_token()) is None


async def test_warm_populates_cache(mock_jwks_client: MockJwksClient, mock_monotonic: MockMonotonicClock) -> None:
    """Warm the cache with one key-set fetch."""
    verifier = build_verifier(jwks_client=mock_jwks_client, clock=mock_monotonic)
    await verifier.warm()
    assert await verifier.verify_token(sign_token(kid="ec-1", key=EC_KEY, algorithm="ES256")) is not None
    assert mock_jwks_client.fetch_count == 1


def _auth(**overrides: Any) -> AuthConfig:
    """Return OAuth settings for the test issuer."""
    values: dict[str, Any] = {"issuer_url": ISSUER, "audience": AUDIENCE, "resource_server_url": AUDIENCE}
    return AuthConfig(**(values | overrides))


def test_key_endpoint_prefers_internal_jwks_url() -> None:
    """Probe the configured JWKS URL, otherwise the issuer's discovery document."""
    internal = "http://keycloak:8080/realms/voice/protocol/openid-connect/certs"
    assert key_endpoint(_auth(jwks_url=internal)) == internal
    assert key_endpoint(_auth()) == f"{ISSUER}/.well-known/openid-configuration"
    assert _auth(jwks_url="").jwks_url is None


async def test_startup_retries_until_provider_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry discovery and the first key fetch while the provider starts, keeping the public issuer."""
    attempts = {"discovery": 0, "keys": 0}

    async def discover(issuer_url: str, *, timeout_seconds: float) -> str:
        """Fail once, then return an internal JWKS URI."""
        attempts["discovery"] += 1
        if attempts["discovery"] == 1:
            raise ConfigurationError("not ready")
        return "http://keycloak:8080/certs"

    async def warm(self: JwtTokenVerifier) -> bool:
        """Report no keys once, then success."""
        attempts["keys"] += 1
        return attempts["keys"] > 1

    monkeypatch.setattr(factory, "discover_jwks_uri", discover)
    monkeypatch.setattr(JwtTokenVerifier, "warm", warm)
    monkeypatch.setattr(factory.asyncio, "sleep", _no_sleep)
    verifier = await factory.build_token_verifier(_auth(startup_timeout_seconds=30))
    assert attempts == {"discovery": 2, "keys": 2}
    assert verifier._issuer == ISSUER


async def test_unreachable_provider_fails_startup_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ConfigurationError once the startup budget is spent instead of serving without key checks."""

    async def discover(issuer_url: str, *, timeout_seconds: float) -> str:
        """Always fail."""
        raise ConfigurationError("connection refused")

    monkeypatch.setattr(factory, "discover_jwks_uri", discover)
    with pytest.raises(ConfigurationError, match="startup timeout"):
        await factory.build_token_verifier(_auth(startup_timeout_seconds=0))


async def _no_sleep(_seconds: float) -> None:
    """Skip retry delays."""
