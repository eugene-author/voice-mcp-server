"""register_voice tool."""

from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app.domain.jobs import RegisterVoiceCommand
from app.mcp_server.dependencies import ToolServices
from app.mcp_server.errors import mapped_errors
from app.mcp_server.tools._common import submit_and_respond


def register(server: MCPServer, services: ToolServices) -> None:
    """Register the register_voice tool."""

    @server.tool(
        name="register_voice",
        description="Register a cloned voice from an authorized reference recording (5-15 s of clean speech).",
    )
    @mapped_errors
    async def register_voice(
        name: Annotated[str, Field(min_length=1, max_length=80)],
        reference_audio_path: Annotated[str, Field(description="Absolute path of the reference recording.")],
        consent_confirmed: Annotated[
            bool, Field(description="Confirm you are authorized to clone this voice; must be true.")
        ],
        reference_text: Annotated[
            str | None, Field(max_length=2000, description="Exact transcript of the recording (recommended).")
        ] = None,
        language: Annotated[str | None, Field(description="ISO 639-1 code of the recording.")] = None,
        wait: Annotated[bool, Field(description="Wait for completion up to the server's inline limit.")] = True,
        request_id: Annotated[str | None, Field(max_length=128, description="Idempotency key.")] = None,
    ) -> dict[str, Any]:
        """Submit a voice registration job."""
        command = RegisterVoiceCommand(
            name=name,
            reference_audio_path=Path(reference_audio_path),
            consent_confirmed=consent_confirmed,
            reference_text=reference_text,
            language=language,
            request_id=request_id,
        )
        return await submit_and_respond(services, command, wait=wait)
