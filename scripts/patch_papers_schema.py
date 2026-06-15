"""
scripts/patch_papers_schema.py

Ensure papers table has filename column + unique constraint.
Run BEFORE migrate_v2.py if papers table already exists without filename.

Run: PYTHONPATH=. python scripts/patch_papers_schema.py
"""
import asyncio
from sqlalchemy import text
from api.core.database import engine


async def patch():
    async with engine.begin() as conn:
        print("Patching papers table schema...")

        await conn.execute(text("""
            ALTER TABLE papers
            ADD COLUMN IF NOT EXISTS filename TEXT
        """))

        # backfill filename from arxiv_id where possible
        await conn.execute(text("""
            UPDATE papers
            SET filename = arxiv_id || '.pdf'
            WHERE filename IS NULL AND arxiv_id IS NOT NULL
        """))

        # add unique constraint on filename if not exists
        r = await conn.execute(text("""
            SELECT constraint_name FROM information_schema.table_constraints
            WHERE table_name = 'papers'
              AND constraint_type = 'UNIQUE'
              AND constraint_name = 'papers_filename_key'
        """))
        if not r.fetchone():
            await conn.execute(text("""
                ALTER TABLE papers
                ADD CONSTRAINT papers_filename_key UNIQUE (filename)
            """))
            print("  ✓ unique constraint on filename added")
        else:
            print("  ✓ unique constraint already exists")

        print("Done.")


if __name__ == "__main__":
    asyncio.run(patch())
