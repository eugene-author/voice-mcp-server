"""Domain error hierarchy with stable codes and retryable flags."""

from typing import ClassVar


class VoiceMcpError(Exception):
    """Base error whose code and message are safe to return to callers."""

    code: ClassVar[str] = "internal_error"
    default_retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, retryable: bool | None = None) -> None:
        """Store the caller-safe message and an optional retryable override."""
        super().__init__(message)
        self.message = message
        self.retryable = self.default_retryable if retryable is None else retryable


class ClaimLostError(Exception):
    """The running claim no longer holds its job; the worker stops it, and this does not reach a caller."""


class InvalidRequestError(VoiceMcpError):
    """Validation failure, unsupported option or path outside the allowlist."""

    code = "invalid_request"


class DuplicateRequestError(InvalidRequestError):
    """Concurrent submission with a request_id that was just used by the same owner."""


class NotFoundError(VoiceMcpError):
    """Missing or not-owned job, voice or asset."""

    code = "not_found"


class CapabilityUnavailableError(VoiceMcpError):
    """Feature whose optional extra is not installed on this host."""

    code = "capability_unavailable"


class ModelNotInstalledError(VoiceMcpError):
    """Model missing for the active backend."""

    code = "model_not_installed"


class BackendUnavailableError(VoiceMcpError):
    """Forced backend that this host cannot run."""

    code = "backend_unavailable"


class MediaError(VoiceMcpError):
    """Corrupt or unsupported audio, or a media tool failure."""

    code = "media_error"


class InferenceError(VoiceMcpError):
    """Model inference failure such as OOM or a kernel error."""

    code = "inference_error"


class AssetPublishError(VoiceMcpError):
    """Failure to publish an asset to durable storage."""

    code = "asset_publish_failed"
    default_retryable = True


class PersistenceUnavailableError(VoiceMcpError):
    """Database unavailable or failing."""

    code = "persistence_unavailable"
    default_retryable = True


class ConfigurationError(VoiceMcpError):
    """Invalid configuration detected at startup."""

    code = "configuration_error"


class InternalError(VoiceMcpError):
    """Unexpected failure whose details are not exposed to callers."""

    code = "internal_error"

    @classmethod
    def from_exception(cls, exc: BaseException) -> InternalError:
        """Return a caller-safe error that names only the exception type."""
        return cls(f"unexpected internal error ({type(exc).__name__})")
