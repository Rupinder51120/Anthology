"""
src/retrieval/retriever.py
pgvector-only retrieval: vector search + PostgreSQL FTS + RRF + Cross-Encoder
"""

import os
import numpy as np
from src.retrieval.embedder import embed_texts

RRF_K = 60


# ── pgvector search ───────────────────────────────────────────────────

async def pgvector_search(query_embedding: list[float], top_k: int, db) -> list[dict]:
    from sqlalchemy import text
    query_vec = "[" + ",".join(str(x) for x in query_embedding) + "]"
    result = await db.execute(text(f"""
        SELECT
            chunk_id, source, title, authors, year,
            section, section_priority, chunk_type, text,
            1 - (embedding <=> '{query_vec}'::vector) as similarity
        FROM chunks
        ORDER BY embedding <=> '{query_vec}'::vector
        LIMIT {top_k}
    """))
    return [
        {
            "text": row.text,
            "metadata": {
                "chunk_id":         row.chunk_id,
                "source":           row.source,
                "title":            row.title,
                "authors":          row.authors or "",
                "year":             row.year,
                "section":          row.section or "",
                "section_priority": row.section_priority or 0.5,
                "chunk_type":       row.chunk_type or "general",
                "rerank_score":     float(row.similarity),
            }
        }
        for row in result.fetchall()
    ]


# ── postgres FTS search ───────────────────────────────────────────────

async def postgres_fts_search(query: str, top_k: int, db) -> list[dict]:
    from sqlalchemy import text
    clean_query = " & ".join(w for w in query.split() if len(w) > 2)
    if not clean_query:
        return []
    result = await db.execute(
        text(f"""
            SELECT
                chunk_id, source, title, authors, year,
                section, section_priority, chunk_type, text,
                ts_rank(to_tsvector('english', text),
                        to_tsquery('english', :query)) as rank
            FROM chunks
            WHERE to_tsvector('english', text) @@ to_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT {top_k}
        """),
        {"query": clean_query}
    )
    return [
        {
            "text": row.text,
            "metadata": {
                "chunk_id":         row.chunk_id,
                "source":           row.source,
                "title":            row.title,
                "authors":          row.authors or "",
                "year":             row.year,
                "section":          row.section or "",
                "section_priority": row.section_priority or 0.5,
                "chunk_type":       row.chunk_type or "general",
                "rerank_score":     float(row.rank),
            }
        }
        for row in result.fetchall()
    ]


# ── RRF fusion ────────────────────────────────────────────────────────

def rrf_fuse(vec_results: list[dict], fts_results: list[dict], top_k: int) -> list[dict]:
    all_chunks = {r["metadata"]["chunk_id"]: r for r in vec_results + fts_results}
    scores = {}
    for rank, r in enumerate(vec_results):
        cid = r["metadata"]["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
    for rank, r in enumerate(fts_results):
        cid = r["metadata"]["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [all_chunks[cid] for cid in top_ids if cid in all_chunks]


# ── cross-encoder reranker ────────────────────────────────────────────

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs  = [(query, c["text"]) for c in chunks]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = [c for _, c in ranked[:top_k]]
        for i, (score, _) in enumerate(ranked[:top_k]):
            result[i]["metadata"]["rerank_score"] = float(score)
        return result
    except Exception:
        return chunks[:top_k]


# ── intent detection ──────────────────────────────────────────────────

def detect_query_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["recommend", "suggest", "find papers", "similar to"]):
        return "recommendation"
    if any(w in q for w in ["compare", "difference", "versus", "vs", "better"]):
        return "comparison"
    if any(w in q for w in ["summarize", "summary", "overview", "explain"]):
        return "explanation"
    return "factual"


# ── main retrieve function ────────────────────────────────────────────

async def retrieve(query: str, top_k: int = 5, db=None) -> list[dict]:
    """
    pgvector + PostgreSQL FTS + RRF + Cross-Encoder.
    db is required — always called from FastAPI with an active session.
    """
    if db is None:
        raise ValueError("db session required for pgvector retrieval")

    query_emb = embed_texts([query])[0].tolist()

    vec_results = await pgvector_search(query_emb, top_k=top_k * 3, db=db)
    fts_results = await postgres_fts_search(query, top_k=top_k * 3, db=db)

    fused = rrf_fuse(vec_results, fts_results, top_k=top_k * 2)
    return rerank(query, fused, top_k=top_k)
