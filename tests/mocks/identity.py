"""JWKS client and token verifier test doubles."""

from typing import Any

import jwt
from mcp.server.auth.provider import AccessToken


class MockJwksClient:
    """Serves signing keys from an in-memory key set and counts fetches."""

    def __init__(self, keys: dict[str, Any]) -> None:
        """Store public keys by key ID."""
        self.keys = dict(keys)
        self.fetch_count = 0

    def get_signing_key(self, kid: str) -> jwt.PyJWK:
        """Return the key for kid, counting a key-set fetch."""
        self.fetch_count += 1
        if kid not in self.keys:
            raise jwt.PyJWKClientError(f"unable to find a signing key that matches {kid}")
        return self._jwk(kid)

    def get_signing_keys(self) -> list[jwt.PyJWK]:
        """Return every key, counting a key-set fetch."""
        self.fetch_count += 1
        return [self._jwk(kid) for kid in self.keys]

    def _jwk(self, kid: str) -> jwt.PyJWK:
        """Wrap a public key as a PyJWK."""
        key = self.keys[kid]
        algorithm = "ES256" if "EC" in type(key).__name__ else "RS256"
        data = jwt.algorithms.get_default_algorithms()[algorithm].to_jwk(key, as_dict=True)
        return jwt.PyJWK({**data, "kid": kid, "alg": algorithm})


class MockTokenVerifier:
    """Accepts tokens from a fixed table; for tests only."""

    def __init__(self, tokens: dict[str, AccessToken]) -> None:
        """Store accepted tokens."""
        self.tokens = tokens

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return the configured access token or None."""
        return self.tokens.get(token)
