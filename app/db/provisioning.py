"""Checks the configured database before migrations, creates it on request, and explains failures in one line."""

import asyncpg
from sqlalchemy.engine import URL, make_url

from app.core.settings import DatabaseSettings
from app.db.engine import plain_database_url
from app.domain.errors import ConfigurationError

DEFAULT_PORT = 5432
MAINTENANCE_DATABASE = "postgres"
LOGIN_ERRORS = (asyncpg.InvalidPasswordError, asyncpg.InvalidAuthorizationSpecificationError)


async def ensure_database(settings: DatabaseSettings, *, create: bool) -> bool:
    """Check the configured database, create it when missing if create is true, and return whether it was created."""
    if settings.url is None:
        raise ConfigurationError("DATABASE_URL is not set")
    url = make_url(plain_database_url(settings.url.get_secret_value()))
    if not url.database:
        raise ConfigurationError("DATABASE_URL does not name a database")
    try:
        await _check(url, settings.connect_timeout_seconds)
    except asyncpg.InvalidCatalogNameError as exc:
        if not create:
            raise ConfigurationError(
                f"database {url.database} does not exist at {_address(url)}; "
                "create it, or run migrate --create-database"
            ) from exc
        return await _create(url, settings.connect_timeout_seconds)
    return False


async def _check(url: URL, timeout_seconds: float) -> None:
    """Open and close one connection to the URL's database."""
    connection = await _connect(url, timeout_seconds)
    await connection.close()


async def _connect(url: URL, timeout_seconds: float) -> asyncpg.Connection:
    """Connect to the URL's database, translating failures other than a missing database."""
    try:
        return await asyncpg.connect(url.render_as_string(hide_password=False), timeout=timeout_seconds)
    except LOGIN_ERRORS as exc:
        raise ConfigurationError(
            f"could not sign in to PostgreSQL at {_address(url)} as {url.username}; check DATABASE_URL"
        ) from exc
    except (OSError, TimeoutError) as exc:
        raise ConfigurationError(
            f"could not reach PostgreSQL at {_address(url)}; check that it is running and check DATABASE_URL"
        ) from exc


async def _create(url: URL, timeout_seconds: float) -> bool:
    """Create the URL's database through the server's maintenance database, and return whether this call created it."""
    connection = await _connect(url.set(database=MAINTENANCE_DATABASE), timeout_seconds)
    try:
        await connection.execute(f"CREATE DATABASE {_quoted(str(url.database))}")
    except asyncpg.DuplicateDatabaseError:
        return False
    except asyncpg.InsufficientPrivilegeError as exc:
        raise ConfigurationError(
            f"{url.username} cannot create database {url.database} at {_address(url)}; "
            "create it, or grant that login CREATEDB"
        ) from exc
    finally:
        await connection.close()
    return True


def _address(url: URL) -> str:
    """Return the URL's host and port, which are safe to show."""
    return f"{url.host}:{url.port or DEFAULT_PORT}"


def _quoted(name: str) -> str:
    """Return a database name quoted as a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'
