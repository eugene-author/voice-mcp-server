"""list_models tool."""

from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from app.domain.models import ModelKind
from app.mcp_server.dependencies import ToolServices
from app.mcp_server.errors import mapped_errors


def register(server: MCPServer, services: ToolServices) -> None:
    """Register the list_models tool."""

    @server.tool(name="list_models", description="List registry models per backend, with installed and active flags.")
    @mapped_errors
    async def list_models(kind: Literal["stt", "tts", "denoise"] | None = None) -> dict[str, Any]:
        """Return model statuses."""
        services.caller_resolver.current()
        statuses = services.models.list_models(ModelKind(kind) if kind else None)
        return {
            "models": [
                {
                    "name": status.name,
                    "kind": str(status.kind),
                    "backend": status.backend,
                    "installed": status.installed,
                    "active": status.active_backend,
                    "loaded": status.loaded,
                    "approx_size_mb": status.approx_size_mb,
                    **({"variant": status.variant} if status.variant else {}),
                }
                for status in statuses
            ]
        }
