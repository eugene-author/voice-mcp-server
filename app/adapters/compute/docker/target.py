"""Docker compute target: inspects, starts and stops one container through a Docker socket proxy."""

import logging
from urllib.parse import quote

import httpx

from app.domain.errors import BackendUnavailableError, ConfigurationError
from app.domain.workers import TargetState

logger = logging.getLogger(__name__)
FORCE_STOP_SECONDS = 10
STATE_BY_DOCKER_STATUS = {
    "created": TargetState.STOPPED,
    "exited": TargetState.STOPPED,
    "dead": TargetState.STOPPED,
    "restarting": TargetState.STARTING,
    "running": TargetState.RUNNING,
    "removing": TargetState.STOPPING,
}
"""Docker State.Status to target state; anything else, such as paused, is unknown."""


def _client(endpoint: str, timeout_seconds: float, transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient:
    """Return a client for unix:///path or http(s)://host:port Docker endpoints."""
    if transport is not None:
        return httpx.AsyncClient(base_url="http://docker", transport=transport, timeout=timeout_seconds)
    if endpoint.startswith("unix://"):
        socket_path = endpoint.removeprefix("unix://")
        return httpx.AsyncClient(
            base_url="http://docker", transport=httpx.AsyncHTTPTransport(uds=socket_path), timeout=timeout_seconds
        )
    if endpoint.startswith(("http://", "https://")):
        return httpx.AsyncClient(base_url=endpoint, timeout=timeout_seconds)
    raise ConfigurationError("gpu_worker.on_demand.docker.endpoint must be unix:///path or http(s)://host:port")


class DockerComputeTarget:
    """Starts and stops the worker container; every call has a timeout and errors read as unknown."""

    def __init__(
        self,
        *,
        endpoint: str,
        container: str,
        stop_timeout_seconds: int,
        request_timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create the HTTP client; stop_timeout_seconds is Docker's grace period before it kills the worker."""
        self._client = _client(endpoint, request_timeout_seconds, transport)
        self._path = f"/containers/{quote(container, safe='')}"
        self._stop_timeout = stop_timeout_seconds
        self._request_timeout = request_timeout_seconds

    async def status(self) -> TargetState:
        """Inspect the container; a missing container or an API error is unknown."""
        try:
            response = await self._client.get(f"{self._path}/json")
        except httpx.HTTPError as exc:
            logger.warning("docker inspect failed", extra={"error": type(exc).__name__})
            return TargetState.UNKNOWN
        if response.status_code != httpx.codes.OK:
            logger.warning("docker inspect failed", extra={"status": response.status_code})
            return TargetState.UNKNOWN
        status = str((response.json().get("State") or {}).get("Status", ""))
        return STATE_BY_DOCKER_STATUS.get(status, TargetState.UNKNOWN)

    async def start(self) -> None:
        """Start the container; already running is fine."""
        await self._post(f"{self._path}/start", within=self._request_timeout)

    async def stop(self, *, force: bool) -> None:
        """Stop the container with the drain grace period, or FORCE_STOP_SECONDS when forced; stopped is fine."""
        grace = FORCE_STOP_SECONDS if force else self._stop_timeout
        await self._post(f"{self._path}/stop", params={"t": grace}, within=grace + self._request_timeout)

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def _post(self, path: str, *, within: float, params: dict[str, int] | None = None) -> None:
        """POST and accept 204 or 304 (no change); raise a retryable error otherwise."""
        try:
            response = await self._client.post(path, params=params, timeout=within)
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(f"docker request failed: {type(exc).__name__}") from exc
        if response.status_code not in (httpx.codes.NO_CONTENT, httpx.codes.NOT_MODIFIED):
            raise BackendUnavailableError(f"docker returned HTTP {response.status_code} for {path}")
