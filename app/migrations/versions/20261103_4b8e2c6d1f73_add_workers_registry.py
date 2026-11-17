"""Add the workers registry used by the api role to see the GPU worker's state and capabilities."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "4b8e2c6d1f73"
down_revision: str | None = "9e3d5b7a1c40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the workers table and its heartbeat index."""
    op.create_table(
        "workers",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("capabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("drain_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("drained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "status IN ('starting', 'ready', 'busy', 'draining', 'stopping', 'stopped')",
            name=op.f("ck_workers_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_workers")),
    )
    op.create_index("ix_workers_heartbeat", "workers", [sa.literal_column("heartbeat_at DESC")], unique=False)


def downgrade() -> None:
    """Drop the workers table."""
    op.drop_index("ix_workers_heartbeat", table_name="workers")
    op.drop_table("workers")
