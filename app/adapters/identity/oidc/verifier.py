"""MCP SDK TokenVerifier backed by PyJWT, with a key-ID cache and throttled key-set refreshes."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import jwt
from mcp.server.auth.provider import AccessToken

from app.ports.clock import MonotonicClock

logger = logging.getLogger(__name__)


class SigningKeySource(Protocol):
    """Subset of PyJWKClient used by the verifier."""

    def get_signing_key(self, kid: str) -> jwt.PyJWK:
        """Return the signing key for a key ID, fetching the key set if needed."""
        ...

    def get_signing_keys(self) -> list[jwt.PyJWK]:
        """Return every signing key of the key set."""
        ...


@dataclass(frozen=True, slots=True)
class _CachedKey:
    """Verified signing key with its expiry on the monotonic clock."""

    key: Any
    """Public key object returned by PyJWK.key."""

    expires_at: float
    """Monotonic time after which the key must be refetched."""


class JwtTokenVerifier:
    """Verifies RS/ES-signed JWT access tokens for one issuer and audience."""

    def __init__(
        self,
        *,
        jwks_client: SigningKeySource,
        issuer: str,
        audience: str,
        algorithms: frozenset[str],
        leeway_seconds: int,
        key_ttl_seconds: int,
        min_refresh_seconds: int,
        clock: MonotonicClock,
    ) -> None:
        """Store verification settings and start with an empty key cache."""
        self._jwks_client = jwks_client
        self._issuer = issuer
        self._audience = audience
        self._algorithms = algorithms
        self._leeway_seconds = leeway_seconds
        self._key_ttl_seconds = key_ttl_seconds
        self._min_refresh_seconds = min_refresh_seconds
        self._clock = clock
        self._key_by_kid: dict[str, _CachedKey] = {}
        self._last_fetch = float("-inf")
        self._refresh_lock = asyncio.Lock()

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return an AccessToken for a valid JWT, otherwise None."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            return None
        algorithm, kid = header.get("alg"), header.get("kid")
        if algorithm not in self._algorithms or not isinstance(kid, str):
            return None
        key = await self._signing_key(kid)
        if key is None:
            return None
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[algorithm],
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway_seconds,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as exc:
            logger.debug("token rejected", extra={"kid": kid, "reason": type(exc).__name__})
            return None
        return self._to_access_token(token, claims)

    async def warm(self) -> bool:
        """Fetch the key set so the first request does not pay for it; return whether any key was loaded."""
        async with self._refresh_lock:
            self._last_fetch = self._clock.monotonic()
            try:
                keys = await asyncio.to_thread(self._jwks_client.get_signing_keys)
            except jwt.PyJWKClientError:
                logger.warning("could not fetch the JWKS key set")
                return False
            expires_at = self._last_fetch + self._key_ttl_seconds
            for signing_key in keys:
                if signing_key.key_id:
                    self._key_by_kid[signing_key.key_id] = _CachedKey(key=signing_key.key, expires_at=expires_at)
        return bool(self._key_by_kid)

    async def _signing_key(self, kid: str) -> Any | None:
        """Return a cached key or fetch it, at most once per refresh interval."""
        cached = self._key_by_kid.get(kid)
        if cached is not None and cached.expires_at > self._clock.monotonic():
            return cached.key
        async with self._refresh_lock:
            now = self._clock.monotonic()
            cached = self._key_by_kid.get(kid)
            if cached is not None and cached.expires_at > now:
                return cached.key
            if now - self._last_fetch < self._min_refresh_seconds:
                return None
            self._last_fetch = now
            try:
                signing_key = await asyncio.to_thread(self._jwks_client.get_signing_key, kid)
            except jwt.PyJWKClientError:
                self._key_by_kid.pop(kid, None)
                return None
            self._key_by_kid[kid] = _CachedKey(key=signing_key.key, expires_at=now + self._key_ttl_seconds)
            return signing_key.key

    def _to_access_token(self, token: str, claims: dict[str, Any]) -> AccessToken:
        """Map claims (scope/scp, azp/client_id, sub, exp) to an AccessToken bound to this resource."""
        scope = claims.get("scope")
        if isinstance(scope, str):
            scopes = scope.split()
        else:
            scp = claims.get("scp")
            scopes = [str(item) for item in scp] if isinstance(scp, list) else []
        client_id = claims.get("azp") or claims.get("client_id") or ""
        expires_at = claims.get("exp")
        return AccessToken(
            token=token,
            client_id=str(client_id),
            scopes=scopes,
            expires_at=int(expires_at) if isinstance(expires_at, int | float) else None,
            resource=self._audience,
            subject=str(claims["sub"]),
            claims={"iss": claims["iss"]},
        )
