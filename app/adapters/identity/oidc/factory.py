"""Builds the JWT token verifier for the configured OIDC provider."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import jwt

from app.adapters.identity.oidc.discovery import discover_jwks_uri
from app.adapters.identity.oidc.verifier import JwtTokenVerifier
from app.core.clock import SystemMonotonicClock
from app.core.settings import AuthConfig
from app.domain.errors import ConfigurationError

logger = logging.getLogger(__name__)
MAX_RETRY_DELAY_SECONDS = 5.0


def key_endpoint(auth: AuthConfig) -> str:
    """Return the identity-provider URL both the verifier and the health check use: auth.jwks_url or discovery."""
    if auth.jwks_url:
        return auth.jwks_url
    if not auth.issuer_url:
        raise ConfigurationError("OAuth requires auth.issuer_url")
    return auth.issuer_url.rstrip("/") + "/.well-known/openid-configuration"


async def build_token_verifier(auth: AuthConfig) -> JwtTokenVerifier:
    """Build the verifier and load its keys, retrying until auth.startup_timeout_seconds, then ConfigurationError."""
    if not auth.issuer_url or not auth.audience:
        raise ConfigurationError("OAuth requires auth.issuer_url and auth.audience")
    issuer_url = auth.issuer_url
    deadline = time.monotonic() + auth.startup_timeout_seconds

    async def resolve() -> str:
        """Return the configured JWKS URL or discover it from the issuer metadata."""
        return auth.jwks_url or await discover_jwks_uri(issuer_url, timeout_seconds=auth.jwks_fetch_timeout_seconds)

    jwks_uri = await _retry_until(deadline, resolve, what="OIDC discovery")
    client = jwt.PyJWKClient(
        jwks_uri,
        cache_jwk_set=True,
        lifespan=auth.jwks_cache_seconds,
        timeout=int(auth.jwks_fetch_timeout_seconds),
    )
    verifier = JwtTokenVerifier(
        jwks_client=client,
        issuer=issuer_url,
        audience=auth.audience,
        algorithms=frozenset(auth.algorithms),
        leeway_seconds=auth.leeway_seconds,
        key_ttl_seconds=auth.jwks_cache_seconds,
        min_refresh_seconds=auth.jwks_min_refresh_seconds,
        clock=SystemMonotonicClock(),
    )
    if auth.warm_jwks_on_startup:

        async def load_keys() -> None:
            """Fetch the key set and raise while it is unavailable or empty."""
            if not await verifier.warm():
                raise ConfigurationError(f"no signing keys could be loaded from {jwks_uri}")

        await _retry_until(deadline, load_keys, what="JWKS key fetch")
    return verifier


async def _retry_until[T](deadline: float, operation: Callable[[], Awaitable[T]], *, what: str) -> T:
    """Run operation until it succeeds, retrying ConfigurationError with growing delays until the deadline."""
    delay = 1.0
    while True:
        try:
            return await operation()
        except ConfigurationError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ConfigurationError(f"{what} did not succeed before the startup timeout: {exc.message}") from exc
            logger.warning("identity provider not ready; retrying", extra={"step": what, "error": exc.message})
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, MAX_RETRY_DELAY_SECONDS)
