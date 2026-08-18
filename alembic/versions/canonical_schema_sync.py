"""canonical schema sync
Revision ID: canonical_sync_001
Revises: enrichment_flag_001
Create Date: 2026-06-29
"""
from alembic import op
import sqlalchemy as sa

revision = 'canonical_sync_001'
down_revision = 'enrichment_flag_001'
branch_labels = None
depends_on = None

def upgrade():
    # 1. Fix papers table: add doi and paper_embedding
    op.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS doi VARCHAR(255)")
    op.execute("ALTER TABLE papers ADD COLUMN IF NOT EXISTS paper_embedding vector(768)")

    # 2. Fix chunks table: increase chunk_id length
    op.execute("ALTER TABLE chunks ALTER COLUMN chunk_id TYPE VARCHAR(32)")

    # 3. Fix queries table: add paper_id
    op.execute("ALTER TABLE queries ADD COLUMN IF NOT EXISTS paper_id UUID")
    # queries.paper_id + its FK were already created by 5890fefb391a's initial
    # create_table (auto-named queries_paper_id_fkey, default ON DELETE NO
    # ACTION). Drop that one before adding the named ON DELETE SET NULL
    # version below -- Postgres enforces every FK constraint on a column, so
    # leaving both in place means the NO ACTION constraint blocks deleting a
    # referenced paper, silently defeating the SET NULL behavior this
    # migration intends. IF EXISTS makes this safe to run against a database
    # where the constraint was never named that way (e.g. re-created by hand).
    op.execute("ALTER TABLE queries DROP CONSTRAINT IF EXISTS queries_paper_id_fkey")
    op.execute("ALTER TABLE queries ADD CONSTRAINT fk_queries_paper FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE SET NULL")

    # 4. Add default to queries.retrieval_mode
    op.execute("ALTER TABLE queries ALTER COLUMN retrieval_mode SET DEFAULT 'hybrid'")



def downgrade():
    op.execute("ALTER TABLE queries DROP CONSTRAINT IF EXISTS fk_queries_paper")
    op.execute("ALTER TABLE queries DROP COLUMN IF EXISTS paper_id")
    op.execute("ALTER TABLE chunks ALTER COLUMN chunk_id TYPE VARCHAR(20)")
    op.execute("ALTER TABLE papers DROP COLUMN IF EXISTS doi")
    op.execute("ALTER TABLE papers DROP COLUMN IF EXISTS paper_embedding")

