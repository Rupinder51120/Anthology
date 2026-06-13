"""add multimodal columns to chunks

Revision ID: multimodal_001
Revises: specter2_768
Create Date: 2026-06-12
"""
from alembic import op

revision = 'multimodal_001'
down_revision = 'specter2_768'
branch_labels = None
depends_on = None

def upgrade():
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_type VARCHAR(20) DEFAULT 'text'")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_path VARCHAR(500)")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_markdown TEXT")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_summary TEXT")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS figure_number VARCHAR(50)")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_number INTEGER")

def downgrade():
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_type")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS image_path")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS table_markdown")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS table_summary")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS figure_number")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS page_number")
