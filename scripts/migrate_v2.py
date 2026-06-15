"""
scripts/migrate_v2.py

Migration: populate papers table from chunk metadata, link all chunks via paper_id.
Zero re-embedding required.

Run: PYTHONPATH=. python scripts/migrate_v2.py
"""
import asyncio
import json
from pathlib import Path
from sqlalchemy import text
from api.core.database import engine


async def migrate():
    print("=" * 60)
    print("Anthology v2 Migration")
    print("=" * 60)

    async with engine.begin() as conn:

        # ── Step 0: ensure paper_id column exists on chunks ──────
        print("\n[0] Ensuring paper_id column exists on chunks...")
        await conn.execute(text("""
            ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS paper_id UUID
            REFERENCES papers(id) ON DELETE CASCADE
        """))
        print("    ✓ paper_id column ready")

        # ── Step 1: load registry for clean metadata ──────────────
        print("\n[1] Loading download_registry.json...")
        registry = {}
        registry_path = Path("data/download_registry.json")
        if registry_path.exists():
            with open(registry_path) as f:
                raw = json.load(f)
            # reindex by filename
            for v in raw.values():
                if "filename" in v:
                    registry[v["filename"]] = v
            print(f"    ✓ {len(registry)} entries in registry")
        else:
            print("    ⚠ registry not found — will use chunk metadata only")

        # ── Step 2: get all unique sources from chunks ────────────
        print("\n[2] Reading unique sources from chunks...")
        result = await conn.execute(text("""
            SELECT DISTINCT
                source,
                MAX(title)   AS title,
                MAX(authors) AS authors,
                MAX(year)    AS year
            FROM chunks
            WHERE source IS NOT NULL
            GROUP BY source
            ORDER BY source
        """))
        sources = result.fetchall()
        print(f"    ✓ {len(sources)} unique papers found in chunks")

        # ── Step 3: insert paper rows ─────────────────────────────
        print("\n[3] Inserting paper rows...")
        inserted = 0
        skipped = 0

        for row in sources:
            filename = row.source
            reg = registry.get(filename, {})

            # prefer registry metadata
            title   = reg.get("title")   or row.title   or filename.replace(".pdf", "")
            authors_raw = reg.get("authors", row.authors or "")
            if isinstance(authors_raw, list):
                authors = ", ".join(authors_raw[:3])
                if len(authors_raw) > 3:
                    authors += " et al."
            else:
                authors = str(authors_raw)

            year_val = reg.get("year") or row.year
            try:
                year = int(year_val) if year_val else None
            except (ValueError, TypeError):
                year = None

            arxiv_id  = reg.get("arxiv_id")
            doi       = reg.get("doi")
            abstract  = reg.get("abstract", "")[:500]
            topic     = reg.get("topic", "")
            url       = reg.get("url", "")

            # upsert — safe to re-run
            res = await conn.execute(text("""
                INSERT INTO papers (
                    filename, title, authors, year,
                    arxiv_id, doi, abstract, topic, url,
                    source, indexed, chunk_count
                ) VALUES (
                    :filename, :title, :authors, :year,
                    :arxiv_id, :doi, :abstract, :topic, :url,
                    'upload', TRUE, 0
                )
                ON CONFLICT (filename) DO UPDATE SET
                    title    = EXCLUDED.title,
                    authors  = EXCLUDED.authors,
                    year     = EXCLUDED.year,
                    arxiv_id = COALESCE(papers.arxiv_id, EXCLUDED.arxiv_id),
                    abstract = COALESCE(NULLIF(papers.abstract,''), EXCLUDED.abstract),
                    topic    = COALESCE(NULLIF(papers.topic,''), EXCLUDED.topic)
                RETURNING id, (xmax = 0) AS was_inserted
            """), {
                "filename": filename,
                "title":    title,
                "authors":  authors,
                "year":     year,
                "arxiv_id": arxiv_id,
                "doi":      doi,
                "abstract": abstract,
                "topic":    topic,
                "url":      url,
            })
            paper_row = res.fetchone()
            if paper_row.was_inserted:
                inserted += 1
            else:
                skipped += 1

        print(f"    ✓ {inserted} inserted, {skipped} updated")

        # ── Step 4: link chunks → papers via paper_id ─────────────
        print("\n[4] Linking chunks to papers via paper_id...")
        res = await conn.execute(text("""
            UPDATE chunks c
            SET paper_id = p.id
            FROM papers p
            WHERE c.source = p.filename
              AND c.paper_id IS NULL
        """))
        linked = res.rowcount
        print(f"    ✓ {linked} chunks linked")

        # ── Step 5: update chunk_count on papers ──────────────────
        print("\n[5] Updating chunk counts...")
        await conn.execute(text("""
            UPDATE papers p
            SET chunk_count = (
                SELECT COUNT(*) FROM chunks c WHERE c.paper_id = p.id
            )
        """))
        print("    ✓ chunk counts updated")

        # ── Step 6: verify ────────────────────────────────────────
        print("\n[6] Verification...")

        r = await conn.execute(text("SELECT COUNT(*) FROM papers"))
        n_papers = r.scalar()

        r = await conn.execute(text("SELECT COUNT(*) FROM chunks WHERE paper_id IS NOT NULL"))
        n_linked = r.scalar()

        r = await conn.execute(text("SELECT COUNT(*) FROM chunks WHERE paper_id IS NULL"))
        n_orphan = r.scalar()

        r = await conn.execute(text("SELECT COUNT(*) FROM chunks"))
        n_total = r.scalar()

        r = await conn.execute(text("""
            SELECT MIN(chunk_count), MAX(chunk_count), AVG(chunk_count)::int
            FROM papers
        """))
        stats = r.fetchone()

        print(f"    papers table:     {n_papers} rows")
        print(f"    chunks linked:    {n_linked} / {n_total}")
        print(f"    orphan chunks:    {n_orphan}")
        print(f"    chunks per paper: min={stats[0]} max={stats[1]} avg={stats[2]}")

        if n_orphan > 0:
            print(f"\n    ⚠ {n_orphan} orphan chunks — likely missing filenames in papers table")

        # ── Step 7: HNSW index ────────────────────────────────────
        print("\n[7] Checking pgvector HNSW index...")
        r = await conn.execute(text("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'chunks'
              AND indexdef ILIKE '%hnsw%'
        """))
        hnsw = r.fetchone()
        if hnsw:
            print(f"    ✓ HNSW index already exists: {hnsw[0]}")
        else:
            print("    Creating HNSW index (this takes ~30s for 12k chunks)...")
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            print("    ✓ HNSW index created")

    print("\n" + "=" * 60)
    print("Migration complete.")
    print("Next: PYTHONPATH=. python scripts/embed_papers.py")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(migrate())
