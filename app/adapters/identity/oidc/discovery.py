"""OIDC discovery of the provider's JWKS URI."""

import httpx

from app.domain.errors import ConfigurationError


async def discover_jwks_uri(issuer_url: str, *, timeout_seconds: float) -> str:
    """Return jwks_uri from the issuer's metadata, requiring the advertised issuer to match exactly."""
    url = issuer_url.rstrip("/") + "/.well-known/openid-configuration"
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()
            metadata = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ConfigurationError(f"OIDC discovery failed for {issuer_url}") from exc
    if metadata.get("issuer") != issuer_url:
        raise ConfigurationError("discovered issuer does not match auth.issuer_url exactly")
    jwks_uri = metadata.get("jwks_uri")
    if not isinstance(jwks_uri, str) or not jwks_uri:
        raise ConfigurationError("OIDC metadata has no jwks_uri")
    return jwks_uri
