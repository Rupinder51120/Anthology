"""relational integrity
Revision ID: relational_001
Revises: multimodal_001
Create Date: 2026-06-22
"""
from alembic import op
import sqlalchemy as sa

revision = 'relational_001'
down_revision = 'multimodal_001'
branch_labels = None
depends_on = None

def upgrade():
    # 1. Fix papers table schema
    op.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS figure_count INTEGER DEFAULT 0")
    op.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS table_count INTEGER DEFAULT 0")
    
    # 2. Add paper_id to chunks
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS paper_id UUID")
    
    # 3. Add Foreign Key with CASCADE
    op.execute("ALTER TABLE chunks ADD CONSTRAINT fk_chunks_paper FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE")

def downgrade():
    op.execute("ALTER TABLE chunks DROP CONSTRAINT IF EXISTS fk_chunks_paper")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS paper_id")
    op.execute("ALTER TABLE papers DROP COLUMN IF EXISTS figure_count")
    op.execute("ALTER TABLE papers DROP COLUMN IF EXISTS table_count")
