"""
src/retrieval/retriever.py
pgvector + FTS + RRF (section-priority weighted) + Cross-Encoder
HyDE disabled — Ollama too slow on CPU for real-time use
"""

import numpy as np
from src.retrieval.embedder import embed_texts

RRF_K = 60

_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


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


async def postgres_fts_search(query: str, top_k: int, db) -> list[dict]:
    from sqlalchemy import text
    result = await db.execute(
        text(f"""
            SELECT
                chunk_id, source, title, authors, year,
                section, section_priority, chunk_type, text,
                ts_rank(to_tsvector('english', text),
                        plainto_tsquery('english', :query)) as rank
            FROM chunks
            WHERE to_tsvector('english', text) @@ plainto_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT {top_k}
        """),
        {"query": query}
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


def rrf_fuse(vec_results: list[dict], fts_results: list[dict], top_k: int) -> list[dict]:
    all_chunks = {r["metadata"]["chunk_id"]: r for r in vec_results + fts_results}
    scores = {}
    for rank, r in enumerate(vec_results):
        cid      = r["metadata"]["chunk_id"]
        priority = r["metadata"].get("section_priority", 0.5)
        scores[cid] = scores.get(cid, 0) + (1 / (RRF_K + rank + 1)) * priority
    for rank, r in enumerate(fts_results):
        cid      = r["metadata"]["chunk_id"]
        priority = r["metadata"].get("section_priority", 0.5)
        scores[cid] = scores.get(cid, 0) + (1 / (RRF_K + rank + 1)) * priority
    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [all_chunks[cid] for cid in top_ids if cid in all_chunks]


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    try:
        ce     = _get_cross_encoder()
        pairs  = [(query, c["text"]) for c in chunks]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = [c for _, c in ranked[:top_k]]
        for i, (score, _) in enumerate(ranked[:top_k]):
            result[i]["metadata"]["rerank_score"] = float(score)
        return result
    except Exception:
        return chunks[:top_k]


async def retrieve(query: str, top_k: int = 5, db=None) -> list[dict]:
    if db is None:
        raise ValueError("db session required for pgvector retrieval")

    query_emb = embed_texts([query])[0].tolist()

    vec_results = await pgvector_search(query_emb, top_k=top_k * 3, db=db)
    fts_results = await postgres_fts_search(query, top_k=top_k * 3, db=db)

    fused = rrf_fuse(vec_results, fts_results, top_k=top_k * 2)
    return rerank(query, fused, top_k=top_k)
