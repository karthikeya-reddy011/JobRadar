"""track the model used for each stored embedding

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-20
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN embedding_model TEXT")
    op.execute("ALTER TABLE profiles ADD COLUMN embedding_model TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE profiles DROP COLUMN embedding_model")
    op.execute("ALTER TABLE jobs DROP COLUMN embedding_model")
