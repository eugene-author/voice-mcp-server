"""MCP server construction with SDK-managed OAuth resource-server settings."""

from mcp.server.auth.provider import TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from app.core.settings import ServerSettings, Settings
from app.mcp_server.dependencies import ToolServices
from app.mcp_server.tools import register_all

INSTRUCTIONS = (
    "Local speech recognition, Qwen3-TTS synthesis and reference-based voice cloning. Audio inputs are host file "
    "paths inside the allowed input directories; results are returned as file locations. Long jobs keep running "
    "after a tool call returns: poll get_job with the job_id."
)
LOOPBACK_ALLOWED_HOSTS = ("127.0.0.1:*", "localhost:*", "[::1]:*")
LOOPBACK_ALLOWED_ORIGINS = ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*")


def transport_security(server: ServerSettings) -> TransportSecuritySettings:
    """Return Host/Origin validation for Streamable HTTP, always on; empty allowlists mean loopback only."""
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(server.allowed_hosts or LOOPBACK_ALLOWED_HOSTS),
        allowed_origins=list(server.allowed_origins or LOOPBACK_ALLOWED_ORIGINS),
    )


def build_mcp_server(settings: Settings, services: ToolServices, token_verifier: TokenVerifier | None) -> MCPServer:
    """Build the MCP server; HTTP transport always uses OAuth, stdio does not."""
    auth_settings: AuthSettings | None = None
    if token_verifier is not None:
        auth_settings = AuthSettings(
            issuer_url=settings.auth.issuer_url,
            resource_server_url=settings.auth.resource_server_url,
            required_scopes=list(settings.auth.required_scopes),
            validate_token_resource=True,
        )
    server = MCPServer(
        name=settings.server.name,
        instructions=INSTRUCTIONS,
        auth=auth_settings,
        token_verifier=token_verifier,
        log_level=settings.observability.log_level,
    )
    register_all(server, services)
    return server
