"""initial schema: companies, jobs, profiles, job_status

Revision ID: 0001
Revises:
Create Date: 2026-10-19
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# One statement per op.execute: psycopg3 does not allow multiple statements in one call
# when parameters are involved, so keep them separate.


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TABLE companies (
          id SERIAL PRIMARY KEY,
          name TEXT NOT NULL,
          ats TEXT NOT NULL,
          slug TEXT NOT NULL,
          UNIQUE (ats, slug)
        )
    """)

    # VECTOR(1536): change to match your embedding model's dimension before you embed anything.
    op.execute("""
        CREATE TABLE jobs (
          id SERIAL PRIMARY KEY,
          source TEXT NOT NULL,
          external_id TEXT NOT NULL,
          company_id INT NOT NULL REFERENCES companies(id),
          title TEXT NOT NULL,
          description TEXT NOT NULL DEFAULT '',
          location TEXT,
          is_remote BOOLEAN NOT NULL DEFAULT FALSE,
          seniority TEXT,
          skills TEXT[],
          apply_url TEXT NOT NULL,
          posted_at TIMESTAMPTZ,
          first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          is_active BOOLEAN NOT NULL DEFAULT TRUE,
          content_hash TEXT,
          embedding VECTOR(1536),
          UNIQUE (source, external_id)
        )
    """)
    op.execute("CREATE INDEX jobs_active_company_idx ON jobs (company_id) WHERE is_active")
    # Not unique on purpose: one company can post the same title in the same city twice.
    op.execute("CREATE INDEX jobs_content_hash_idx ON jobs (content_hash)")
    # Add an HNSW index on `embedding` in Phase 4, after embeddings exist.

    op.execute("""
        CREATE TABLE profiles (
          id SERIAL PRIMARY KEY,
          skills TEXT[],
          years_experience NUMERIC,
          titles TEXT[],
          locations TEXT[],
          remote_ok BOOLEAN,
          seniority TEXT,
          embedding VECTOR(1536)
        )
    """)

    # ON DELETE CASCADE makes the Phase 6 "delete my data" button a single DELETE on profiles.
    op.execute("""
        CREATE TABLE job_status (
          profile_id INT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          job_id INT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
          status TEXT NOT NULL,
          PRIMARY KEY (profile_id, job_id)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS job_status")
    op.execute("DROP TABLE IF EXISTS profiles")
    op.execute("DROP TABLE IF EXISTS jobs")
    op.execute("DROP TABLE IF EXISTS companies")
