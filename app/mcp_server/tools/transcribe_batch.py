"""transcribe_batch tool."""

from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app.domain.jobs import BatchTranscribeCommand
from app.mcp_server.dependencies import ToolServices
from app.mcp_server.errors import mapped_errors
from app.mcp_server.tools._common import submit_and_respond


def register(server: MCPServer, services: ToolServices) -> None:
    """Register the transcribe_batch tool."""

    @server.tool(name="transcribe_batch", description="Transcribe every audio file in an allowed directory.")
    @mapped_errors
    async def transcribe_batch(
        directory: Annotated[str, Field(description="Absolute directory path on the server host.")],
        recursive: bool = False,
        preset: Annotated[str | None, Field(description="fast, balanced, accurate or studio.")] = None,
        wait: Annotated[bool, Field(description="Wait for the whole batch up to the inline limit.")] = False,
        request_id: Annotated[str | None, Field(max_length=128, description="Idempotency key.")] = None,
    ) -> dict[str, Any]:
        """Submit a batch transcription."""
        command = BatchTranscribeCommand(
            directory=Path(directory), recursive=recursive, preset=preset, request_id=request_id
        )
        return await submit_and_respond(services, command, wait=wait)
