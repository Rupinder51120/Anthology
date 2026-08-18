"""add missing chunks indexes (content_type, paper_id+section_priority, section)

Revision ID: chunks_idx_001
Revises: canonical_sync_001
Create Date: 2026-08-17

These three indexes exist on the real running database (confirmed via
backup_pre_remediation.sql) but were never captured in any migration -- they
were added out-of-band. retriever.py filters/orders on content_type, section,
and paper_id+section_priority (RRF section-priority weighting), so a fresh
database built via `alembic upgrade head` needs them to match the schema the
application actually expects. `ix_chunks_source` already exists from the
5890fefb391a migration and is not duplicated here.
"""
from alembic import op

revision = 'chunks_idx_001'
down_revision = 'canonical_sync_001'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON chunks (content_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper_priority ON chunks (paper_id, section_priority DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks (section)")


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_chunks_content_type")
    op.execute("DROP INDEX IF EXISTS idx_chunks_paper_priority")
    op.execute("DROP INDEX IF EXISTS idx_chunks_section")
