"""profiles: add name (handle) and updated_at

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-20
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # `name` is a short handle ("me", "karthik", "friend1"). The Phase 3 eval file refers to
    # profiles by this name, so it has to be unique.
    op.execute("ALTER TABLE profiles ADD COLUMN name TEXT NOT NULL DEFAULT 'me'")
    op.execute("ALTER TABLE profiles ALTER COLUMN name DROP DEFAULT")
    op.execute("ALTER TABLE profiles ADD CONSTRAINT profiles_name_key UNIQUE (name)")
    op.execute("ALTER TABLE profiles ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT now()")


def downgrade() -> None:
    op.execute("ALTER TABLE profiles DROP COLUMN updated_at")
    op.execute("ALTER TABLE profiles DROP CONSTRAINT profiles_name_key")
    op.execute("ALTER TABLE profiles DROP COLUMN name")
