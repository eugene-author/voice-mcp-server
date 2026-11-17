"""Add jobs.claim_token, the identity of the claim allowed to change a running job."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7c1d9e3a5b20"
down_revision: str | None = "4b8e2c6d1f73"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable column; running rows keep NULL, match no token, and recovery resolves them when they go stale."""
    op.add_column("jobs", sa.Column("claim_token", sa.UUID(), nullable=True))


def downgrade() -> None:
    """Drop the column, which removes claim fencing."""
    op.drop_column("jobs", "claim_token")
