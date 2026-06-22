"""add enrichment flag to chunks
Revision ID: enrichment_flag_001
Revises: relational_001
Create Date: 2026-06-22
"""
from alembic import op
import sqlalchemy as sa

revision = 'enrichment_flag_001'
down_revision = 'relational_001'
branch_labels = None
depends_on = None

def upgrade():
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_enriched BOOLEAN DEFAULT false")

def downgrade():
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS is_enriched")
