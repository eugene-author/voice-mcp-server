"""list_voices tool."""

from typing import Any

from mcp.server.mcpserver import MCPServer

from app.mcp_server.dependencies import ToolServices
from app.mcp_server.errors import mapped_errors


def register(server: MCPServer, services: ToolServices) -> None:
    """Register the list_voices tool."""

    @server.tool(name="list_voices", description="List preset speakers and your cloned voices.")
    @mapped_errors
    async def list_voices(language: str | None = None) -> dict[str, Any]:
        """Return voice summaries."""
        voices = await services.voices.list_voices(owner=services.caller_resolver.current(), language=language)
        return {
            "voices": [
                {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "kind": str(voice.kind),
                    "status": str(voice.status),
                    **({"language": voice.language} if voice.language else {}),
                }
                for voice in voices
            ]
        }
