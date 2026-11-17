"""Structured logging to stderr, keeping stdout free for stdio MCP frames."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_STANDARD_ATTRIBUTES = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}
_SECRET_MARKERS = ("token", "authorization", "password", "secret", "database_url", "presign")


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Return the record's extra fields, dropping any whose name suggests a secret."""
    return {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_ATTRIBUTES and not any(marker in key.lower() for marker in _SECRET_MARKERS)
    }


class JsonFormatter(logging.Formatter):
    """One JSON object per line with extra fields included."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the record as JSON."""
        payload: dict[str, Any] = {
            "at": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **_extra_fields(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable lines with extra fields appended."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the record as one readable line."""
        extras = " ".join(f"{key}={value}" for key, value in _extra_fields(record).items())
        line = f"{self.formatTime(record)} {record.levelname:<7} {record.name}: {record.getMessage()} {extras}".rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(*, level: str, log_format: str) -> None:
    """Send all logging to stderr in the configured format."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if log_format == "json" else ConsoleFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "urllib3", "botocore", "numba", "faster_whisper"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLevelName(level)))
