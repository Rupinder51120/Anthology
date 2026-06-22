import asyncio
import asyncpg
import os
from datetime import datetime

async def backfill():
    db_url = os.getenv("DATABASE_URL", "postgresql://anthology:anthology@localhost:5432/anthology")
    db_url_pg = db_url.replace("postgresql+asyncpg://", "postgresql://").replace("+asyncpg", "")
    
    conn = await asyncpg.connect(db_url_pg)
    print("Connected to database. Starting backfill...")

    # 1. Get all unique sources from chunks
    sources = await conn.fetch("SELECT DISTINCT source FROM chunks")
    print(f"Found {len(sources)} unique paper sources.")

    for record in sources:
        filename = record['source']
        
        # 2. Create or find Paper record
        paper_id = await conn.fetchval("""
            INSERT INTO papers (filename, title, authors, year, topic, url, chunk_count, indexed, created_at, updated_at)
            VALUES ($1, 'Unknown Title', 'Unknown', NULL, NULL, NULL, 0, false, now(), now())
            ON CONFLICT (filename) DO UPDATE SET updated_at = now()
            RETURNING id
        """, filename)
        
        # 3. Link chunks to this paper
        await conn.execute("UPDATE chunks SET paper_id = $1 WHERE source = $2", paper_id, filename)
        
        # 4. Update Paper stats
        stats = await conn.fetchrow("""
            SELECT 
                count(*) as total,
                count(*) FILTER (WHERE content_type = 'figure') as figs,
                count(*) FILTER (WHERE content_type = 'table') as tbls
            FROM chunks WHERE paper_id = $1
        """, paper_id)
        
        await conn.execute("""
            UPDATE papers SET 
                chunk_count = $1, figure_count = $2, table_count = $3, indexed = true 
            WHERE id = $4
        """, stats['total'], stats['figs'], stats['tbls'], paper_id)
        
        print(f"Backfilled: {filename} -> {paper_id}")

    await conn.close()
    print("\nBackfill complete. All orphans linked.")

if __name__ == "__main__":
    asyncio.run(backfill())
