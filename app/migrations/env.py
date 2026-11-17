"""Alembic environment: async migrations against DATABASE_URL from Settings."""

import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from app.core.settings import load_settings
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.engine import create_engine

target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    """Run migrations on one synchronous connection."""
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def _run_online() -> None:
    """Connect with the application's engine settings and migrate."""
    settings = load_settings(server={"transport": "stdio"})
    engine = create_engine(settings.database)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_configure)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    raise SystemExit("offline SQL generation is not supported; run against a database")
asyncio.run(_run_online())
