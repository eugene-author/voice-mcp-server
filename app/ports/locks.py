"""Single-worker lock port."""

from typing import Protocol


class WorkerLock(Protocol):
    """Process-lifetime lock that allows only one inference worker."""

    async def acquire(self) -> bool:
        """Try to take the lock without waiting; return whether it is held."""
        ...

    async def release(self) -> None:
        """Release the lock if held."""
        ...
