"""Maps the authenticated request to a Caller: the SDK access token over HTTP, the OS user over stdio."""

from mcp.server.auth.middleware.auth_context import get_access_token

from app.domain.caller import LOCAL_ISSUER, Caller
from app.domain.errors import InvalidRequestError


class CallerResolver:
    """Returns the Caller of the current tool call on the active transport."""

    def __init__(self, *, trusted_local: Caller | None) -> None:
        """Store the stdio caller; None means every call must carry an OAuth token."""
        self._local = trusted_local

    @classmethod
    def for_stdio(cls, os_user: str) -> CallerResolver:
        """Return a resolver whose callers are the local OS user."""
        return cls(trusted_local=Caller(LOCAL_ISSUER, os_user))

    def current(self) -> Caller:
        """Return the caller of the current tool call."""
        token = get_access_token()
        if token is not None:
            issuer = (token.claims or {}).get("iss")
            if not issuer or not token.subject:
                raise InvalidRequestError("access token lacks issuer or subject")
            return Caller(issuer=str(issuer), subject=token.subject, client_id=token.client_id or None)
        if self._local is None:
            raise InvalidRequestError("authentication required")
        return self._local
