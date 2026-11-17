"""speech_to_text tool."""

from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app.domain.jobs import TranscribeCommand
from app.mcp_server.dependencies import ToolServices
from app.mcp_server.errors import mapped_errors
from app.mcp_server.schemas.inputs import SttOptionsInput
from app.mcp_server.tools._common import submit_and_respond


def register(server: MCPServer, services: ToolServices) -> None:
    """Register the speech_to_text tool."""

    @server.tool(name="speech_to_text", description="Transcribe an audio file from an allowed input directory.")
    @mapped_errors
    async def speech_to_text(
        file_path: Annotated[str, Field(description="Absolute path of the audio file on the server host.")],
        model: Annotated[str | None, Field(description="Whisper model: tiny, base, small, medium, large-v3.")] = None,
        language: Annotated[str | None, Field(description="ISO 639-1 code, or auto to detect.")] = None,
        preset: Annotated[str | None, Field(description="fast, balanced, accurate or studio.")] = None,
        wait: Annotated[bool, Field(description="Wait for completion up to the server's inline limit.")] = True,
        options: SttOptionsInput | None = None,
        request_id: Annotated[str | None, Field(max_length=128, description="Idempotency key.")] = None,
    ) -> dict[str, Any]:
        """Submit a transcription job."""
        command = TranscribeCommand(
            input_path=Path(file_path),
            model=model,
            language=language,
            preset=preset,
            options=options.to_domain() if options else None,
            request_id=request_id,
        )
        return await submit_and_respond(services, command, wait=wait)
