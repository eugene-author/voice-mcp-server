"""Waiting for state that background tasks change, with a deadline."""

import asyncio
import itertools
from collections.abc import Callable

POLL_SECONDS = 0.005


async def eventually(condition: Callable[[], object], *, within: float = 5.0) -> None:
    """Return once condition() is truthy; fail with TimeoutError after within seconds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    for _ in itertools.count():
        if condition():
            return
        if loop.time() >= deadline:
            raise TimeoutError("condition not met in time")
        await asyncio.sleep(POLL_SECONDS)
