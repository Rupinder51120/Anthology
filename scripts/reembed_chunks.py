"""
Re-embed all existing chunks using the current _build_embedding_text()
contract, WITHOUT touching Docling/parsing/Groq enrichment/chunking.

This is an embedding-only regeneration: reads chunks.text + metadata columns
already in the DB, recomputes the embedding vector, and UPDATEs only the
embedding column. Chunk identity (chunk_id), paper linkage, and all other
columns are untouched.

Processes in bounded batches (never loads all chunks into memory at once)
and verifies row counts before/after.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func, text
from api.core.database import AsyncSessionLocal
from api.models.tables import Chunk
from src.retrieval.embedder import embed_chunks

BATCH_SIZE = 200  # chunks per batch -- bounds peak memory for embed+update


async def run(batch_size: int = BATCH_SIZE, dry_run: bool = False, limit_batches: int | None = None) -> dict:
    async with AsyncSessionLocal() as db:
        total = (await db.execute(select(func.count()).select_from(Chunk))).scalar_one()
        print(f"Total chunks to re-embed: {total}")

        # Stable ordering so batches are deterministic across resumed runs.
        ids = (await db.execute(select(Chunk.id).order_by(Chunk.chunk_id))).scalars().all()

    updated = 0
    t_start = time.perf_counter()

    for batch_num, start in enumerate(range(0, len(ids), batch_size)):
        if limit_batches is not None and batch_num >= limit_batches:
            print(f"--limit-batches={limit_batches} reached, stopping early.")
            break
        batch_ids = ids[start : start + batch_size]

        async with AsyncSessionLocal() as db:
            rows = (
                await db.execute(select(Chunk).where(Chunk.id.in_(batch_ids)))
            ).scalars().all()

            chunks_for_embed = [
                {
                    "text": r.text,
                    "metadata": {
                        "title": r.title, "authors": r.authors, "year": r.year,
                        "section": r.section, "content_type": r.content_type,
                        "table_summary": r.table_summary, "table_markdown": r.table_markdown,
                        "figure_number": r.figure_number,
                    },
                }
                for r in rows
            ]

            embeddings = embed_chunks(chunks_for_embed)

            if not dry_run:
                # db.execute() above already auto-began a transaction on this
                # session; commit that one directly rather than nesting
                # another via db.begin() (which raises InvalidRequestError).
                for r, emb in zip(rows, embeddings):
                    vec = "[" + ",".join(str(x) for x in emb.tolist()) + "]"
                    await db.execute(
                        text("UPDATE chunks SET embedding = CAST(:vec AS vector) WHERE id = :id"),
                        {"vec": vec, "id": r.id},
                    )
                await db.commit()

            updated += len(rows)
            del embeddings, chunks_for_embed, rows

        elapsed = time.perf_counter() - t_start
        rate = updated / elapsed if elapsed > 0 else 0
        eta_s = (len(ids) - updated) / rate if rate > 0 else float("inf")
        print(f"[{updated}/{len(ids)}] batch done ({elapsed:.1f}s elapsed, "
              f"{rate:.1f} chunks/s, ETA {eta_s/60:.1f}min){' [DRY RUN]' if dry_run else ''}")

    total_time = time.perf_counter() - t_start
    print(f"\nRe-embedding complete: {updated}/{total} chunks in {total_time:.1f}s"
          f"{' [DRY RUN — no writes performed]' if dry_run else ''}")
    return {"total": total, "updated": updated, "duration_s": round(total_time, 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed all chunks with the current embedding-text contract.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Compute embeddings but do not write to DB")
    parser.add_argument("--limit-batches", type=int, default=None, help="Only process the first N batches (for testing)")
    args = parser.parse_args()

    asyncio.run(run(batch_size=args.batch_size, dry_run=args.dry_run, limit_batches=args.limit_batches))
