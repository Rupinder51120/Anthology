"""
src/retrieval/retriever.py
Cloud-optimized: PostgreSQL FTS + RRF scoring. No ML models loaded at runtime.
Embeddings stored in DB but query embedding skipped on free tier.
"""

RRF_K = 60


async def postgres_fts_search(query: str, top_k: int, db) -> list[dict]:
    from sqlalchemy import text
    clean_query = " & ".join(w for w in query.split() if len(w) > 2)
    if not clean_query:
        clean_query = query.split()[0] if query.split() else "research"
    try:
        result = await db.execute(
            text(f"""
                SELECT
                    chunk_id, source, title, authors, year,
                    section, section_priority, chunk_type, text,
                    ts_rank_cd(to_tsvector('english', text),
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
    except Exception:
        return []


async def postgres_semantic_fallback(query: str, top_k: int, db) -> list[dict]:
    """Fallback: return recent high-quality chunks matching any query term."""
    from sqlalchemy import text
    terms = [w for w in query.split() if len(w) > 3][:5]
    if not terms:
        return []
    conditions = " OR ".join(f"text ILIKE '%{t}%'" for t in terms)
    try:
        result = await db.execute(text(f"""
            SELECT chunk_id, source, title, authors, year,
                   section, section_priority, chunk_type, text,
                   quality_score as rank
            FROM chunks
            WHERE {conditions}
            ORDER BY quality_score DESC
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
                    "rerank_score":     float(row.rank) if row.rank else 0.0,
                }
            }
            for row in result.fetchall()
        ]
    except Exception:
        return []


def rrf_fuse(list1: list[dict], list2: list[dict], top_k: int) -> list[dict]:
    all_chunks = {r["metadata"]["chunk_id"]: r for r in list1 + list2}
    scores = {}
    for rank, r in enumerate(list1):
        cid = r["metadata"]["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
    for rank, r in enumerate(list2):
        cid = r["metadata"]["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [all_chunks[cid] for cid in top_ids if cid in all_chunks]


def detect_query_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["recommend", "suggest", "find papers", "similar to"]):
        return "recommendation"
    if any(w in q for w in ["compare", "difference", "versus", "vs", "better"]):
        return "comparison"
    if any(w in q for w in ["summarize", "summary", "overview", "explain"]):
        return "explanation"
    return "factual"


async def retrieve(query: str, top_k: int = 5, db=None) -> list[dict]:
    """
    PostgreSQL FTS retrieval — no ML models, fits in 512MB RAM.
    Primary: plainto_tsquery FTS
    Fallback: ILIKE term matching
    Fusion: RRF
    """
    if db is None:
        raise ValueError("db session required")

    fts_results = await postgres_fts_search(query, top_k=top_k * 3, db=db)

    if len(fts_results) < top_k:
        fallback = await postgres_semantic_fallback(query, top_k=top_k * 2, db=db)
        fused = rrf_fuse(fts_results, fallback, top_k=top_k)
    else:
        fused = fts_results[:top_k]

    return fused
