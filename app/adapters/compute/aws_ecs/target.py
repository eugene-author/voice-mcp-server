"""Amazon ECS compute target: scales one GPU worker service between desired count."""

import asyncio
import logging
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.domain.errors import BackendUnavailableError
from app.domain.workers import TargetState

logger = logging.getLogger(__name__)
CLIENT_CONFIG = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 3, "mode": "standard"})


def state_of(*, desired: int, running: int, pending: int) -> TargetState:
    """Map an ECS service's counts to a target state."""
    if desired <= 0:
        return TargetState.STOPPING if running or pending else TargetState.STOPPED
    return TargetState.RUNNING if running > 0 else TargetState.STARTING


class EcsComputeTarget:
    """Starts and stops the worker by setting the service's desired count; credentials come from the default chain."""

    def __init__(self, *, region: str, cluster: str, service: str, client: Any | None = None) -> None:
        """Create the ECS client for the region unless one is given."""
        self._client = client or boto3.client("ecs", region_name=region, config=CLIENT_CONFIG)
        self._cluster = cluster
        self._service = service

    async def status(self) -> TargetState:
        """Describe the service; an API error or a missing service is unknown."""
        try:
            response = await asyncio.to_thread(
                self._client.describe_services, cluster=self._cluster, services=[self._service]
            )
        except (BotoCoreError, ClientError) as exc:
            logger.warning("ecs describe_services failed", extra={"error": type(exc).__name__})
            return TargetState.UNKNOWN
        services = response.get("services") or []
        if not services:
            logger.warning("ecs service not found", extra={"failures": len(response.get("failures") or [])})
            return TargetState.UNKNOWN
        service = services[0]
        return state_of(
            desired=int(service.get("desiredCount", 0)),
            running=int(service.get("runningCount", 0)),
            pending=int(service.get("pendingCount", 0)),
        )

    async def start(self) -> None:
        """Set the desired count to 1."""
        await self._scale(1)

    async def stop(self, *, force: bool) -> None:
        """Set the desired count to 0; force is the same call, since ECS applies the task's stopTimeout."""
        await self._scale(0)

    async def aclose(self) -> None:
        """Nothing to release; boto3 clients need no closing."""

    async def _scale(self, desired: int) -> None:
        """Update the service's desired count."""
        try:
            await asyncio.to_thread(
                self._client.update_service, cluster=self._cluster, service=self._service, desiredCount=desired
            )
        except (BotoCoreError, ClientError) as exc:
            raise BackendUnavailableError(f"ecs update_service failed: {type(exc).__name__}") from exc
