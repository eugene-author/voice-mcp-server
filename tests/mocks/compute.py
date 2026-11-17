"""Compute target and ECS client doubles."""

from collections.abc import Callable
from typing import Any

from app.domain.workers import TargetState


class MockComputeTarget:
    """Records start and stop requests; its state is set by the test."""

    def __init__(self, state: TargetState = TargetState.STOPPED) -> None:
        """Start in the given state."""
        self.state = state
        self.calls: list[str] = []
        self.fail_status = False
        self.fail_start = False
        self.on_stop: Callable[[], None] | None = None

    async def status(self) -> TargetState:
        """Return the configured state, or raise when asked to."""
        if self.fail_status:
            raise ConnectionError("platform API unreachable")
        return self.state

    async def start(self) -> None:
        """Record a start and raise when asked to; otherwise the target begins starting."""
        self.calls.append("start")
        if self.fail_start:
            raise ConnectionError("platform refused the start")
        self.state = TargetState.STARTING

    async def stop(self, *, force: bool) -> None:
        """Record a stop, let the test spend time on it, and begin stopping."""
        self.calls.append("force_stop" if force else "stop")
        if self.on_stop is not None:
            self.on_stop()
        self.state = TargetState.STOPPING

    async def aclose(self) -> None:
        """Nothing to release."""


class MockEcsClient:
    """Stands in for a boto3 ECS client: describe_services and update_service on one service."""

    def __init__(self, *, desired: int = 0, running: int = 0, pending: int = 0) -> None:
        """Start with the given counts."""
        self.counts = {"desiredCount": desired, "runningCount": running, "pendingCount": pending}
        self.updates: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.missing = False

    def describe_services(self, *, cluster: str, services: list[str]) -> dict[str, Any]:
        """Return the service's counts, nothing when missing, or raise the configured error."""
        if self.error is not None:
            raise self.error
        if self.missing:
            return {"services": [], "failures": [{"arn": services[0], "reason": "MISSING"}]}
        return {"services": [{"serviceName": services[0], "clusterArn": cluster, **self.counts}], "failures": []}

    def update_service(self, *, cluster: str, service: str, desiredCount: int) -> dict[str, Any]:
        """Record the update and apply the desired count."""
        if self.error is not None:
            raise self.error
        self.updates.append({"cluster": cluster, "service": service, "desiredCount": desiredCount})
        self.counts["desiredCount"] = desiredCount
        return {"service": {"serviceName": service, **self.counts}}
