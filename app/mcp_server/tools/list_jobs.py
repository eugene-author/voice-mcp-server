"""list_jobs tool."""

from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app.domain.jobs import JobFilter, JobStatus, JobType
from app.mcp_server.dependencies import ToolServices
from app.mcp_server.errors import mapped_errors
from app.mcp_server.schemas.responses import job_summary
from app.mcp_server.tools._common import parse_job_id

StatusName = Literal[
    "queued", "running", "retry_wait", "interrupted", "completed", "completed_with_errors", "failed", "cancelled"
]
TypeName = Literal["speech_to_text", "text_to_speech", "transcribe_batch", "register_voice"]


def register(server: MCPServer, services: ToolServices) -> None:
    """Register the list_jobs tool."""

    @server.tool(name="list_jobs", description="List your jobs, newest first, with cursor pagination.")
    @mapped_errors
    async def list_jobs(
        status: StatusName | None = None,
        type: TypeName | None = None,
        parent_job_id: str | None = None,
        limit: Annotated[int | None, Field(ge=1, le=100)] = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Return one page of job summaries."""
        job_filter = JobFilter(
            status=JobStatus(status) if status else None,
            type=JobType(type) if type else None,
            parent_job_id=parse_job_id(parent_job_id) if parent_job_id else None,
        )
        page = await services.queries.list(
            owner=services.caller_resolver.current(), job_filter=job_filter, limit=limit, cursor=cursor
        )
        return {"items": [job_summary(view) for view in page.items], "next_cursor": page.next_cursor}
