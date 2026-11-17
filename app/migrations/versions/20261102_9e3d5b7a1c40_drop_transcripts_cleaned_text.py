"""Drop transcripts.cleaned_text, a column that no code ever wrote or read.

Refuses to run if any row has a value, so data that some external process may have stored is not discarded
silently; in that case export it and drop the column deliberately.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9e3d5b7a1c40"
down_revision: str | None = "7c1f0e4b9a21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the unused column after checking that it holds no data."""
    populated = op.get_bind().scalar(sa.text("SELECT count(*) FROM transcripts WHERE cleaned_text IS NOT NULL"))
    if populated:
        raise RuntimeError(
            f"transcripts.cleaned_text holds data in {populated} rows; export it before dropping the column"
        )
    op.drop_column("transcripts", "cleaned_text")


def downgrade() -> None:
    """Restore the empty nullable column."""
    op.add_column("transcripts", sa.Column("cleaned_text", sa.Text(), nullable=True))
