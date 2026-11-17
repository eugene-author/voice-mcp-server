"""PostgreSQL fixtures: a migrated test database, truncated between tests."""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.settings import DatabaseSettings
from app.db.engine import create_engine, create_session_factory
from app.db.unit_of_work import SqlUnitOfWorkFactory

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://voice:voice@127.0.0.1:5433/voice_test")
ROOT = Path(__file__).resolve().parents[2]
TABLES = ", ".join(
    (
        "jobs",
        "job_events",
        "media_assets",
        "transcripts",
        "transcript_segments",
        "transcript_words",
        "voices",
        "voice_versions",
        "quality_metrics",
        "workers",
    )
)


def alembic_config() -> Config:
    """Return the project's Alembic configuration."""
    return Config(str(ROOT / "alembic.ini"))


async def set_claim_token_column(engine: AsyncEngine, change: str) -> None:
    """Alter the jobs table so the current ORM can insert rows into a schema older than claim_token."""
    async with engine.begin() as connection:
        await connection.execute(text(f"ALTER TABLE jobs {change}"))


@pytest.fixture(scope="session")
def migrated_database() -> Iterator[str]:
    """Upgrade the test database to head once per session."""
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    command.upgrade(alembic_config(), "head")
    yield TEST_DATABASE_URL
    if previous is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = previous


@pytest.fixture
async def engine(migrated_database: str) -> AsyncIterator[AsyncEngine]:
    """Return an engine on a clean, migrated database."""
    engine = create_engine(DatabaseSettings(url=migrated_database))  # type: ignore[arg-type]
    async with engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {TABLES} CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
def sql_uow(engine: AsyncEngine) -> SqlUnitOfWorkFactory:
    """Return a SQL unit-of-work factory."""
    return SqlUnitOfWorkFactory(create_session_factory(engine))
