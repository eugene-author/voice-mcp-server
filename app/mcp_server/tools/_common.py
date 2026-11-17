"""Helpers shared by the job-submitting tools."""

import uuid
from typing import Any

from app.domain.errors import InvalidRequestError
from app.domain.ids import JobId
from app.domain.jobs import JobCommand
from app.mcp_server.dependencies import ToolServices
from app.mcp_server.schemas.responses import job_envelope


def parse_job_id(raw: str) -> JobId:
    """Return a job ID or raise invalid_request."""
    try:
        return JobId(uuid.UUID(raw))
    except ValueError as exc:
        raise InvalidRequestError("job_id must be a UUID") from exc


async def submit_and_respond(services: ToolServices, command: JobCommand, *, wait: bool) -> dict[str, Any]:
    """Submit a command, optionally wait up to the inline limit, and return the job envelope."""
    owner = services.caller_resolver.current()
    submission = await services.jobs.submit(command, owner=owner)
    timeout = services.inline_wait_seconds if wait else 0.0
    view = await services.jobs.wait_for(submission.job_id, owner=owner, timeout_seconds=timeout)
    envelope = job_envelope(view)
    if submission.voice_id is not None:
        envelope["voice_id"] = submission.voice_id
    return envelope
