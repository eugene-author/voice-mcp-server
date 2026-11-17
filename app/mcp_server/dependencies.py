"""Services the tool handlers call, built in the composition root."""

from dataclasses import dataclass

from app.mcp_server.auth import CallerResolver
from app.services.health_check_service import HealthCheckService
from app.services.job_query_service import JobQueryService
from app.services.job_service import JobService
from app.services.model_service import ModelService
from app.services.voice_service import VoiceService


@dataclass(frozen=True, slots=True)
class ToolServices:
    """Everything a tool handler may use."""

    caller_resolver: CallerResolver
    jobs: JobService
    queries: JobQueryService
    models: ModelService
    voices: VoiceService
    health: HealthCheckService
    inline_wait_seconds: float
