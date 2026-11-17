"""The single mapping from domain errors to MCP tool error results."""

import functools
import json
import logging
from collections.abc import Awaitable, Callable

from mcp_types import CallToolResult, TextContent

from app.domain.errors import InternalError, VoiceMcpError

logger = logging.getLogger(__name__)


def error_result(error: VoiceMcpError) -> CallToolResult:
    """Return an is_error result whose text and structured content are {error: {code, message}}."""
    payload = {"error": {"code": error.code, "message": error.message}}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))], structured_content=payload, is_error=True
    )


def mapped_errors[**P, R](handler: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R | CallToolResult]]:
    """Wrap a tool handler so failures become structured error results; it is the task boundary for tools."""

    @functools.wraps(handler)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | CallToolResult:
        """Run the handler and translate failures without exposing tracebacks."""
        try:
            return await handler(*args, **kwargs)
        except VoiceMcpError as exc:
            logger.info("tool rejected request", extra={"tool": handler.__name__, "code": exc.code})
            return error_result(exc)
        except Exception as exc:
            logger.exception("tool failed unexpectedly", extra={"tool": handler.__name__})
            return error_result(InternalError.from_exception(exc))

    return wrapper
