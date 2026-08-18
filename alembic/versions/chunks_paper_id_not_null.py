"""enforce chunks.paper_id NOT NULL

Revision ID: chunks_paper_id_nn_001
Revises: session_tables_001
Create Date: 2026-08-17

api/models/tables.py::Chunk.paper_id declares nullable=False, but no prior
migration ever added the NOT NULL constraint at the DB level (relational_001
only added the column + FK, both nullable). Historical orphan chunks were
already repaired out-of-band (scripts/backfill_orphans.py, scripts/migrate_v2.py)
per the Anthology audit (docs/ANTHOLOGY_FULL_AUDIT.md, BUG-09), but nothing
stopped new orphans from being created since. This migration closes that gap.

Safety: verified against the real database before writing this migration --
`SELECT count(*) FROM chunks WHERE paper_id IS NULL` returned 0 (of 12021
total chunks). The guard below re-checks this at migration time and raises
loudly instead of silently deleting/truncating any row that would violate
the constraint, in case orphans appear between audit time and upgrade time.
"""
from alembic import op

revision = 'chunks_paper_id_nn_001'
down_revision = 'session_tables_001'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    orphan_count = conn.exec_driver_sql(
        "SELECT count(*) FROM chunks WHERE paper_id IS NULL"
    ).scalar()
    if orphan_count:
        raise RuntimeError(
            f"Refusing to add NOT NULL constraint: {orphan_count} chunk(s) "
            "have paper_id IS NULL. Investigate and repair (e.g. via "
            "scripts/backfill_orphans.py) before re-running this migration. "
            "This migration will not delete data to force the constraint."
        )
    op.execute("ALTER TABLE chunks ALTER COLUMN paper_id SET NOT NULL")


def downgrade():
    op.execute("ALTER TABLE chunks ALTER COLUMN paper_id DROP NOT NULL")
