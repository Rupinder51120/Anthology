
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import text
from api.core.database import AsyncSessionLocal
from src.retrieval.embedder import embed_texts

load_dotenv()

async def deep_audit():
    query = "How do recent advancements in MLLMs contribute to their overall performance?"
    gt_id = "efd1447db499ba75"

    # We need the query embedding
    query_emb = embed_texts([query])[0].tolist()
    query_vec = "[" + ",".join(str(x) for x in query_emb) + "]"

    async with AsyncSessionLocal() as db:
        # --- Dense Deep Dive ---
        dense_sql = text(f"""
            SELECT chunk_id, 1 - (embedding <=> CAST(:vec AS vector)) as sim
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT 100
        """)
        dense_res = await db.execute(dense_sql, {"vec": query_vec})
        dense_chunks = [row.chunk_id for row in dense_res.fetchall()]

        # --- Sparse Deep Dive ---
        sparse_sql = text(f"""
            SELECT chunk_id, ts_rank_cd(to_tsvector('english', text), websearch_to_tsquery('english', regexp_replace(:query, '\\s+', ' OR ', 'g'))) as sim
            FROM chunks
            WHERE to_tsvector('english', text) @@ websearch_to_tsquery('english', regexp_replace(:query, '\\s+', ' OR ', 'g'))
            ORDER BY sim DESC
            LIMIT 100
        """)
        sparse_res = await db.execute(sparse_sql, {"query": query})
        sparse_chunks = [row.chunk_id for row in sparse_res.fetchall()]

        dense_rank = (dense_chunks.index(gt_id) + 1) if gt_id in dense_chunks else "MISS"
        sparse_rank = (sparse_chunks.index(gt_id) + 1) if gt_id in sparse_chunks else "MISS"

        print(f"Query: {query}")
        print(f"GT ID: {gt_id}")
        print(f"Dense Rank (Top 100): {dense_rank}")
        print(f"Sparse Rank (Top 100): {sparse_rank}")

if __name__ == "__main__":
    asyncio.run(deep_audit())
