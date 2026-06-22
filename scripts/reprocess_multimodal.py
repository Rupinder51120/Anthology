import asyncio
import asyncpg
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.figure_captioner import caption_figure
from src.ingestion.table_summarizer import summarize_table

async def reprocess():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable is not set.")
        sys.exit(1)
    db_url_pg = db_url.replace("postgresql+asyncpg://", "postgresql://").replace("+asyncpg", "")
    
    conn = await asyncpg.connect(db_url_pg)
    print("Connected to database. Scanning for unenriched content...")

    # 1. Find chunks that need enrichment
    # We focus on figures and tables that are not yet enriched
    rows = await conn.fetch("""
        SELECT id, content_type, image_path, table_markdown, figure_number, source, title 
        FROM chunks 
        WHERE is_enriched = false AND content_type IN ('figure', 'table')
    """)
    
    print(f"Found {len(rows)} chunks requiring enrichment.")
    
    success_count = 0
    
    for row in rows:
        chunk_id = row['id']
        ct = row['content_type']
        
        try:
            if ct == 'figure' and row['image_path']:
                # Recover figure caption
                res = caption_figure(row['image_path'], row['title'], row['figure_number'] or "Figure")
                caption = res["caption"]
                
                # Simple heuristic: if caption is just a fallback, don't mark as enriched
                is_fallback = "image not available" in caption or "from the research paper" in caption
                
                if not is_fallback:
                    await conn.execute(
                        "UPDATE chunks SET text = $1, is_enriched = true WHERE id = $2",
                        caption, chunk_id
                    )
                    success_count += 1
                    print(f"Enriched figure: {chunk_id}")

            elif ct == 'table' and row['table_markdown']:
                # Recover table summary
                summary = summarize_table(row['table_markdown'], row['title'])
                if summary:
                    await conn.execute(
                        "UPDATE chunks SET text = text || '\n\nSummary: ' || $1, is_enriched = true WHERE id = $2",
                        summary, chunk_id
                    )
                    success_count += 1
                    print(f"Enriched table: {chunk_id}")
                    
        except Exception as e:
            print(f"Failed to reprocess chunk {chunk_id}: {e}")

    await conn.close()
    print(f"\nReprocessing complete. Successfully enriched {success_count}/{len(rows)} chunks.")

if __name__ == "__main__":
    asyncio.run(reprocess())
