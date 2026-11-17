"""Async engine and session factory, built once in the composition root."""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.settings import DatabaseSettings
from app.domain.errors import ConfigurationError

ASYNC_DRIVER_PREFIX = "postgresql+asyncpg://"


def async_database_url(raw_url: str) -> str:
    """Return the URL with the asyncpg driver, accepting plain postgresql:// URLs."""
    for prefix in ("postgresql://", "postgres://"):
        if raw_url.startswith(prefix):
            return ASYNC_DRIVER_PREFIX + raw_url.removeprefix(prefix)
    if not raw_url.startswith(ASYNC_DRIVER_PREFIX):
        raise ConfigurationError("DATABASE_URL must be a PostgreSQL URL")
    return raw_url


def plain_database_url(raw_url: str) -> str:
    """Return the URL as plain postgresql://, as asyncpg.connect expects for dedicated LISTEN/NOTIFY connections."""
    return "postgresql://" + async_database_url(raw_url).removeprefix(ASYNC_DRIVER_PREFIX)


def create_engine(settings: DatabaseSettings) -> AsyncEngine:
    """Create the process-wide engine with pre-ping, bounded pool and connect/statement timeouts."""
    if settings.url is None:
        raise ConfigurationError("DATABASE_URL is not set")
    return create_async_engine(
        async_database_url(settings.url.get_secret_value()),
        pool_pre_ping=True,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        hide_parameters=True,
        connect_args={
            "timeout": settings.connect_timeout_seconds,
            "server_settings": {"statement_timeout": str(settings.statement_timeout_ms)},
        },
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory whose objects stay usable after commit."""
    return async_sessionmaker(engine, expire_on_commit=False)
