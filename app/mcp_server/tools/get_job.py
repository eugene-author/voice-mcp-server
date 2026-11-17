"""get_job tool."""

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app.mcp_server.dependencies import ToolServices
from app.mcp_server.errors import mapped_errors
from app.mcp_server.schemas.responses import job_envelope
from app.mcp_server.tools._common import parse_job_id


def register(server: MCPServer, services: ToolServices) -> None:
    """Register the get_job tool."""

    @server.tool(name="get_job", description="Return a job's status and, when finished, its output files.")
    @mapped_errors
    async def get_job(
        job_id: str,
        include_download_urls: Annotated[
            bool, Field(description="S3 mode only: include short-lived presigned download URLs.")
        ] = False,
    ) -> dict[str, Any]:
        """Return the job envelope."""
        details = await services.queries.details(
            parse_job_id(job_id), owner=services.caller_resolver.current(), include_download_urls=include_download_urls
        )
        return job_envelope(details.view, download_urls=details.download_urls)
