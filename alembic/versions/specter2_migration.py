"""change embedding dim to 768 for SPECTER2

Revision ID: specter2_768
Revises: 5890fefb391a
Create Date: 2025-01-01
"""
from alembic import op

revision = 'specter2_768'
down_revision = '5890fefb391a'
branch_labels = None
depends_on = None

def upgrade():
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(768)")

def downgrade():
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024)")
