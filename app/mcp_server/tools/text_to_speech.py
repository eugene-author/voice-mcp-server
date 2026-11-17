"""text_to_speech tool."""

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app.domain.jobs import SynthesizeCommand
from app.mcp_server.dependencies import ToolServices
from app.mcp_server.errors import mapped_errors
from app.mcp_server.schemas.inputs import TtsOptionsInput
from app.mcp_server.tools._common import submit_and_respond


def register(server: MCPServer, services: ToolServices) -> None:
    """Register the text_to_speech tool."""

    @server.tool(name="text_to_speech", description="Synthesize speech with a cloned or preset Qwen3-TTS voice.")
    @mapped_errors
    async def text_to_speech(
        text: Annotated[str, Field(min_length=1, description="Text to speak.")],
        voice_id: Annotated[str, Field(description="A voice_id from list_voices.")],
        language: Annotated[str | None, Field(description="ISO 639-1 code, or auto.")] = None,
        preset: Annotated[str | None, Field(description="fast, balanced, accurate or studio.")] = None,
        wait: Annotated[bool, Field(description="Wait for completion up to the server's inline limit.")] = True,
        options: TtsOptionsInput | None = None,
        request_id: Annotated[str | None, Field(max_length=128, description="Idempotency key.")] = None,
    ) -> dict[str, Any]:
        """Submit a synthesis job."""
        command = SynthesizeCommand(
            text=text,
            voice_id=voice_id,
            language=language,
            preset=preset,
            options=options.to_domain() if options else None,
            request_id=request_id,
        )
        return await submit_and_respond(services, command, wait=wait)
