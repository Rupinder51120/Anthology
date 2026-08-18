"""add missing DB-level defaults for created_at/updated_at columns

Revision ID: timestamp_defaults_001
Revises: chunks_paper_id_nn_001
Create Date: 2026-08-18

Discovered during Phase 11 (live API smoke tests) of the backend-completion
pass: every ORM model declares `server_default=func.now()` for
created_at/updated_at, but only `chunks.created_at` actually has that
default at the DB level (papers/chunks insert via raw SQL with an explicit
`now()` literal, so they never depended on it -- everything else does).

Confirmed via information_schema that `queries`, `chat_messages`,
`collections`, `feedback`, and `research_sessions` are all missing the
default. Live-verified impact: `POST /api/v1/collections` and
`POST /api/v1/sessions` both return 500 (NotNullViolationError on
created_at) against the real database -- Collections and Sessions/Chat
creation have been silently broken. The `queries` table has zero rows
ever recorded, meaning query audit-logging has never worked (the failure
happens during response teardown, after the HTTP response is already
sent, so it never surfaces to the client -- see BUG list in
docs/ANTHOLOGY_FULL_AUDIT.md).

This migration only ADDS column defaults -- it does not alter, validate,
or touch any existing row, so it is safe to apply to a database with
live data.
"""
from alembic import op

revision = 'timestamp_defaults_001'
down_revision = 'chunks_paper_id_nn_001'
branch_labels = None
depends_on = None

_TABLES_WITH_CREATED_AT = ["queries", "chat_messages", "collections", "feedback", "research_sessions"]
_TABLES_WITH_UPDATED_AT = ["collections", "research_sessions"]
# papers.updated_at is deliberately excluded -- ingest_service.py always sets
# it explicitly via raw SQL (now()) on both insert and update, so a DB-level
# default isn't needed there and adding one wouldn't change any behavior.


def upgrade():
    for table in _TABLES_WITH_CREATED_AT:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN created_at SET DEFAULT now()")
    for table in _TABLES_WITH_UPDATED_AT:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN updated_at SET DEFAULT now()")


def downgrade():
    for table in _TABLES_WITH_UPDATED_AT:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN updated_at DROP DEFAULT")
    for table in _TABLES_WITH_CREATED_AT:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN created_at DROP DEFAULT")
