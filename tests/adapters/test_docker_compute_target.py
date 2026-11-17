"""Docker compute target over httpx.MockTransport: state mapping, errors, start and stop grace periods."""

import httpx
import pytest

from app.adapters.compute.docker.target import FORCE_STOP_SECONDS, DockerComputeTarget
from app.domain.errors import BackendUnavailableError, ConfigurationError
from app.domain.workers import TargetState

CONTAINER = "voice-mcp-gpu-worker"


def _target(handler: object, *, stop_timeout: int = 1800) -> DockerComputeTarget:
    """Return a target whose HTTP traffic goes to handler."""
    return DockerComputeTarget(
        endpoint="unix:///var/run/docker-proxy.sock",
        container=CONTAINER,
        stop_timeout_seconds=stop_timeout,
        request_timeout_seconds=5,
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


def _inspect(status: str) -> httpx.Response:
    """Return a container inspect response with State.Status."""
    return httpx.Response(200, json={"Name": f"/{CONTAINER}", "State": {"Status": status}})


@pytest.mark.parametrize(
    ("docker_status", "state"),
    [
        ("created", TargetState.STOPPED),
        ("exited", TargetState.STOPPED),
        ("dead", TargetState.STOPPED),
        ("restarting", TargetState.STARTING),
        ("running", TargetState.RUNNING),
        ("removing", TargetState.STOPPING),
        ("paused", TargetState.UNKNOWN),
    ],
)
async def test_container_states_map_to_target_states(docker_status: str, state: TargetState) -> None:
    """Map every Docker State.Status to the documented target state."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Answer the inspect call."""
        seen.append(f"{request.method} {request.url.path}")
        return _inspect(docker_status)

    target = _target(handler)
    assert await target.status() is state
    assert seen == [f"GET /containers/{CONTAINER}/json"]
    await target.aclose()


@pytest.mark.parametrize(
    "response",
    [httpx.Response(404, json={"message": "No such container"}), httpx.Response(500), httpx.Response(403)],
)
async def test_api_errors_read_as_unknown(response: httpx.Response) -> None:
    """Report unknown for a missing container, a server error or a proxy refusal."""
    assert await _target(lambda _request: response).status() is TargetState.UNKNOWN


async def test_unreachable_proxy_reads_as_unknown() -> None:
    """Report unknown when the socket proxy cannot be reached or times out."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail like a dead proxy."""
        raise httpx.ConnectError("connection refused", request=request)

    assert await _target(handler).status() is TargetState.UNKNOWN


async def test_start_and_stop_use_the_drain_grace_period() -> None:
    """Start the container, stop with t=drain timeout, and with t=FORCE_STOP_SECONDS when forced."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Accept start and stop."""
        calls.append(f"{request.method} {request.url.path}?{request.url.query.decode()}")
        return httpx.Response(204)

    target = _target(handler, stop_timeout=1800)
    await target.start()
    await target.stop(force=False)
    await target.stop(force=True)
    assert calls == [
        f"POST /containers/{CONTAINER}/start?",
        f"POST /containers/{CONTAINER}/stop?t=1800",
        f"POST /containers/{CONTAINER}/stop?t={FORCE_STOP_SECONDS}",
    ]


async def test_already_started_or_stopped_is_not_an_error() -> None:
    """Accept 304 Not Modified from start and stop."""
    target = _target(lambda _request: httpx.Response(304))
    await target.start()
    await target.stop(force=False)


async def test_refused_start_raises() -> None:
    """Raise a retryable error when Docker refuses an action."""
    with pytest.raises(BackendUnavailableError, match="HTTP 403"):
        await _target(lambda _request: httpx.Response(403)).start()


def test_endpoint_must_be_a_socket_or_http_url() -> None:
    """Reject endpoints that are neither unix:// nor http(s)://."""
    with pytest.raises(ConfigurationError, match="endpoint"):
        DockerComputeTarget(
            endpoint="tcp://docker:2375", container=CONTAINER, stop_timeout_seconds=1, request_timeout_seconds=1
        )
