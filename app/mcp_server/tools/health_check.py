"""health_check tool."""

from typing import Any

from mcp.server.mcpserver import MCPServer

from app.mcp_server.dependencies import ToolServices
from app.mcp_server.errors import mapped_errors


def register(server: MCPServer, services: ToolServices) -> None:
    """Register the health_check tool."""

    @server.tool(name="health_check", description="Report server status, platform profile, runtimes and devices.")
    @mapped_errors
    async def health_check() -> dict[str, Any]:
        """Return the health report."""
        services.caller_resolver.current()
        report = await services.health.check()
        return {"status": report.status, **report.details}
