"""Single-thread executor that runs all native inference."""

import asyncio
import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class InferenceRunner:
    """Runs blocking inference on one dedicated thread, keeping model state thread-affine and GPU use serial."""

    def __init__(self) -> None:
        """Create the single-worker executor."""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    async def run[T](self, fn: Callable[..., T], /, *args: object, **kwargs: object) -> T:
        """Run fn on the inference thread and await its result."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(fn, *args, **kwargs))

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting work and, unless wait is false, wait for the running call to finish."""
        self._executor.shutdown(wait=wait, cancel_futures=True)
