"""One module per MCP tool."""

from mcp.server.mcpserver import MCPServer

from app.mcp_server.dependencies import ToolServices
from app.mcp_server.tools import (
    get_job,
    health_check,
    list_jobs,
    list_models,
    list_voices,
    register_voice,
    speech_to_text,
    text_to_speech,
    transcribe_batch,
)

TOOL_MODULES = (
    speech_to_text,
    text_to_speech,
    transcribe_batch,
    list_models,
    list_voices,
    register_voice,
    get_job,
    list_jobs,
    health_check,
)


def register_all(server: MCPServer, services: ToolServices) -> None:
    """Register all nine tools."""
    for module in TOOL_MODULES:
        module.register(server, services)
