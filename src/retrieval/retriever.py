"""
src/retrieval/retriever.py

Supports two retrieval modes:
- File-based (FAISS + BM25) — local development
- pgvector + PostgreSQL FTS — cloud deployment

Set USE_PGVECTOR=true in .env to use PostgreSQL.
"""

import os
import numpy as np
from pathlib import Path

# ── embedder ──────────────────────────────────────────────────────────
from src.retrieval.embedder import embed_texts

USE_PGVECTOR = os.getenv("USE_PGVECTOR", "false").lower() == "true"

# ── lazy-loaded indexes (file-based mode) ─────────────────────────────
_chunks     = None
_bm25_index = None
_faiss_vecs = None


def load_chunks(path: str = "indexes/chunks_metadata.json") -> list[dict]:
    global _chunks
    if _chunks is None:
        import json
        with open(path) as f:
            _chunks = json.load(f)
    return _chunks


def get_bm25_index(chunks: list[dict] = None):
    global _bm25_index
    if _bm25_index is None:
        import pickle
        bm25_path = Path("indexes/bm25_index.pkl")
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                _bm25_index = pickle.load(f)
    return _bm25_index


def get_faiss_index():
    global _faiss_vecs
    if _faiss_vecs is None:
        emb_path = Path("indexes/chunk_embeddings.npy")
        if emb_path.exists():
            _faiss_vecs = np.load(str(emb_path))
    return _faiss_vecs


# ── file-based search ─────────────────────────────────────────────────

def faiss_search(query_embedding: np.ndarray, top_k: int = 10) -> list[int]:
    vecs = get_faiss_index()
    if vecs is None:
        return []
    scores = vecs @ query_embedding
    return list(np.argsort(scores)[::-1][:top_k])


def bm25_search(query: str, chunks: list[dict], top_k: int = 10) -> list[int]:
    bm25 = get_bm25_index(chunks)
    if bm25 is None:
        return []
    from rank_bm25 import BM25Okapi
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    return list(np.argsort(scores)[::-1][:top_k])


def reciprocal_rank_fusion(
    list1: list[int], list2: list[int], k: int = 60
) -> list[int]:
    scores = {}
    for rank, idx in enumerate(list1):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(list2):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def deduplicate_candidates(chunks: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for c in chunks:
        key = c["metadata"].get("chunk_id") or c["text"][:50]
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


def boost_by_section_priority(chunks: list[dict]) -> list[dict]:
    return sorted(
        chunks,
        key=lambda c: c["metadata"].get("section_priority", 0.5),
        reverse=True,
    )


def get_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    ce = get_cross_encoder()
    if ce is None or not chunks:
        return chunks[:top_k]
    pairs  = [(query, c["text"]) for c in chunks]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result = [c for _, c in ranked[:top_k]]
    for i, (score, _) in enumerate(ranked[:top_k]):
        result[i]["metadata"]["rerank_score"] = float(score)
    return result


# ── pgvector search ───────────────────────────────────────────────────

async def pgvector_search(
    query_embedding: list[float],
    top_k: int = 10,
    db=None,
) -> list[dict]:
    """Search using pgvector similarity — cloud mode."""
    from sqlalchemy import text

    query_vec = "[" + ",".join(str(x) for x in query_embedding) + "]"
    result = await db.execute(
        text(f"""
            SELECT
                chunk_id, source, title, authors, year,
                section, section_priority, chunk_type, text,
                1 - (embedding <=> '{query_vec}'::vector) as similarity
            FROM chunks
            ORDER BY embedding <=> '{query_vec}'::vector
            LIMIT {top_k}
        """)
    )
    rows = result.fetchall()
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
        for row in rows
    ]


async def postgres_fts_search(
    query: str,
    top_k: int = 10,
    db=None,
) -> list[dict]:
    """Full-text search using PostgreSQL — replaces BM25 in cloud mode."""
    from sqlalchemy import text

    # Clean query for FTS
    clean_query = " & ".join(
        w for w in query.split()
        if len(w) > 2
    )

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
    rows = result.fetchall()
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
        for row in rows
    ]


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

def retrieve(
    query:   str,
    top_k:   int  = 5,
    use_hyde: bool = False,
    db=None,
) -> list[dict]:
    """
    Main retrieval function.
    - If USE_PGVECTOR=true and db provided: uses pgvector + PostgreSQL FTS
    - Otherwise: uses file-based FAISS + BM25
    """
    import asyncio

    if USE_PGVECTOR and db is not None:
        # Cloud mode — pgvector
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_retrieve_pgvector(query, top_k, db))
    else:
        # Local mode — file-based
        return _retrieve_files(query, top_k, use_hyde)


async def _retrieve_pgvector(query: str, top_k: int, db) -> list[dict]:
    """Retrieve using pgvector + PostgreSQL FTS."""
    query_emb = embed_texts([query])[0].tolist()

    # Vector search
    vec_results = await pgvector_search(query_emb, top_k=top_k * 3, db=db)

    # FTS search
    fts_results = await postgres_fts_search(query, top_k=top_k * 3, db=db)

    # Merge by chunk_id using RRF
    vec_ids = [r["metadata"]["chunk_id"] for r in vec_results]
    fts_ids = [r["metadata"]["chunk_id"] for r in fts_results]

    # Build lookup
    all_chunks = {r["metadata"]["chunk_id"]: r for r in vec_results + fts_results}

    # RRF scoring
    k = 60
    scores = {}
    for rank, cid in enumerate(vec_ids):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, cid in enumerate(fts_ids):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)

    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [all_chunks[cid] for cid in top_ids if cid in all_chunks]


def _retrieve_files(query: str, top_k: int, use_hyde: bool) -> list[dict]:
    """Retrieve using file-based FAISS + BM25."""
    chunks = load_chunks()

    # HyDE
    if use_hyde:
        try:
            from src.retrieval.hyde import expand_query_with_hyde
            expanded = expand_query_with_hyde(query)
            query_for_embed = expanded.get("hyde_docs", [query])[0]
        except Exception:
            query_for_embed = query
    else:
        query_for_embed = query

    # Embed
    query_emb = embed_texts([query_for_embed])[0]

    # FAISS
    faiss_ids = faiss_search(query_emb, top_k=top_k * 5)
    # BM25
    bm25_ids  = bm25_search(query, chunks, top_k=top_k * 5)

    # Fuse
    fused_ids = reciprocal_rank_fusion(faiss_ids, bm25_ids)[:top_k * 3]
    candidates = [chunks[i] for i in fused_ids if i < len(chunks)]

    # Dedup + boost
    candidates = deduplicate_candidates(candidates)
    candidates = boost_by_section_priority(candidates)

    return candidates[:top_k]
