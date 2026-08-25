"""add missing DB-level default for collection_papers.added_at

Revision ID: coll_papers_added_at_001
Revises: timestamp_defaults_001
Create Date: 2026-08-25

`timestamp_defaults_001` fixed the missing server-side default for
created_at/updated_at on queries/chat_messages/collections/feedback/
research_sessions, but missed `collection_papers.added_at` -- same root
cause (table created out-of-band in `session_tables_001` with a plain
`TIMESTAMP ... NOT NULL`, no `DEFAULT now()`), same failure mode: the ORM
model (`api/models/tables.py::CollectionPaper.added_at`) declares
`server_default=func.now()`, but nothing enforces that at the DB level,
so a bare `INSERT INTO collection_papers (collection_id, paper_id)`
raises `NotNullViolationError` on `added_at`.

Confirmed live during the Phase 2 browser audit: `POST
/api/v1/collections/{id}/papers/{paper_id}` returned `{"success": true}`
/ 200 while the underlying INSERT failed at commit time (after the
response was already sent) -- the paper was never actually added to the
collection.

This migration only ADDS a column default -- it does not alter, validate,
or touch any existing row, so it is safe to apply to a database with live
data.
"""
from alembic import op

revision = 'coll_papers_added_at_001'
down_revision = 'timestamp_defaults_001'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE collection_papers ALTER COLUMN added_at SET DEFAULT now()")


def downgrade():
    op.execute("ALTER TABLE collection_papers ALTER COLUMN added_at DROP DEFAULT")
