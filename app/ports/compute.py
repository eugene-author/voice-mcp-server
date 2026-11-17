"""Compute target port: starts and stops the compute that runs the GPU worker."""

from typing import Protocol

from app.domain.workers import TargetState

__all__ = ["ComputeTarget", "TargetState"]


class ComputeTarget(Protocol):
    """Starts and stops the compute that runs the GPU worker."""

    async def status(self) -> TargetState:
        """Return the target's current state; unknown when the platform cannot be asked."""
        ...

    async def start(self) -> None:
        """Request the worker to start; idempotent."""
        ...

    async def stop(self, *, force: bool) -> None:
        """Request the worker to stop; force skips any grace period the platform allows skipping."""
        ...

    async def aclose(self) -> None:
        """Release clients held by the target."""
        ...
