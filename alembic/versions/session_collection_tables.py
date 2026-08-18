"""add research_sessions, chat_messages, collections, collection_papers tables

Revision ID: session_tables_001
Revises: chunks_idx_001
Create Date: 2026-08-17

These four tables back the Sessions/Chat and Collections features (see
api/models/tables.py::ResearchSession/ChatMessage/Collection/CollectionPaper
and api/routers/sessions.py, api/routers/collections.py) and are actively
used in production, but were never created by any prior migration -- like
chunks.embedding, they were added out-of-band directly against the running
database. `IF NOT EXISTS` is used throughout so this migration is a safe
no-op against a database that already has them (e.g. one that was hand
-patched before this migration existed), and creates them correctly on a
genuinely fresh database.
"""
from alembic import op

revision = 'session_tables_001'
down_revision = 'chunks_idx_001'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS research_sessions (
            id UUID PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id UUID PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            id UUID PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            color VARCHAR(20) NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS collection_papers (
            collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            added_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            PRIMARY KEY (collection_id, paper_id)
        )
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS collection_papers")
    op.execute("DROP TABLE IF EXISTS collections")
    op.execute("DROP TABLE IF EXISTS chat_messages")
    op.execute("DROP TABLE IF EXISTS research_sessions")
