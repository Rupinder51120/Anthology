"""
src/retrieval/retriever.py
SPECTER2 + pgvector + FTS + modality-boosted RRF + Cohere rerank
FIX: SQL injection removed — all queries use bound parameters
"""

import os
import numpy as np
import cohere
from src.retrieval.embedder import embed_texts
from api.core.models import COHERE_RERANK_MODEL, CROSS_ENCODER_MODEL
from api.core.config import get_settings

settings = get_settings()
RRF_K = 60

_cross_encoder = None

MODALITY_BOOST = {
    "figure":   ["figure", "diagram", "architecture", "shows", "image", "chart", "plot", "visualization"],
    "table":    ["table", "results", "comparison", "benchmark", "scores", "metrics", "performance"],
    "equation": ["equation", "formula", "loss", "objective", "proof"],
}


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def _detect_modality_boost(query: str) -> dict[str, float]:
    q = query.lower()
    boosts = {"text": 1.0, "table": 1.0, "figure": 1.0, "equation": 1.0}
    for modality, signals in MODALITY_BOOST.items():
        if any(s in q for s in signals):
            boosts[modality] = 1.5
    return boosts


async def pgvector_search(
    query_embedding: list[float],
    top_k: int,
    db,
    content_type: str | None = None,
    paper_id: str | None = None,
) -> list[dict]:
    from sqlalchemy import text

    query_vec = "[" + ",".join(str(x) for x in query_embedding) + "]"

    filters = "WHERE embedding IS NOT NULL"
    params: dict = {"vec": query_vec, "k": top_k}
    if content_type:
        filters += " AND content_type = :ct"
        params["ct"] = content_type
    if paper_id:
        filters += " AND paper_id = :pid"
        params["pid"] = paper_id

    sql = text("""
        SELECT
            chunk_id, source, title, authors, year,
            section, section_priority, chunk_type, content_type,
            text, page_number, figure_number, image_path,
            table_markdown, table_summary,
            1 - (embedding <=> CAST(:vec AS vector)) as similarity
        FROM chunks
        {filters}
        ORDER BY embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """.format(filters=filters))
    result = await db.execute(sql, params)
    return [_row_to_dict(r) for r in result.fetchall()]


async def postgres_fts_search(
    query: str,
    top_k: int,
    db,
    content_type: str | None = None,
    paper_id: str | None = None,
) -> list[dict]:
    from sqlalchemy import text

    filters = "WHERE to_tsvector('english', text) @@ plainto_tsquery('english', :query)"
    params: dict = {"query": query, "k": top_k}
    if content_type:
        filters += " AND content_type = :ct"
        params["ct"] = content_type
    if paper_id:
        filters += " AND paper_id = :pid"
        params["pid"] = paper_id

    sql = text("""
        SELECT
            chunk_id, source, title, authors, year,
            section, section_priority, chunk_type, content_type,
            text, page_number, figure_number, image_path,
            table_markdown, table_summary,
            ts_rank(to_tsvector('english', text),
                    plainto_tsquery('english', :query)) as similarity
        FROM chunks
        {filters}
        ORDER BY similarity DESC
        LIMIT :k
    """.format(filters=filters))
    result = await db.execute(sql, params)
    return [_row_to_dict(r) for r in result.fetchall()]


def rrf_fuse(
    vec_results: list[dict],
    fts_results: list[dict],
    top_k: int,
    modality_boosts: dict = None,
) -> list[dict]:
    all_chunks = {r["metadata"]["chunk_id"]: r for r in vec_results + fts_results}
    scores = {}
    for rank, r in enumerate(vec_results):
        cid      = r["metadata"]["chunk_id"]
        priority = r["metadata"].get("section_priority", 0.5)
        mboost   = (modality_boosts or {}).get(r["metadata"].get("content_type", "text"), 1.0)
        scores[cid] = scores.get(cid, 0) + (1 / (RRF_K + rank + 1)) * priority * mboost
    for rank, r in enumerate(fts_results):
        cid      = r["metadata"]["chunk_id"]
        priority = r["metadata"].get("section_priority", 0.5)
        mboost   = (modality_boosts or {}).get(r["metadata"].get("content_type", "text"), 1.0)
        scores[cid] = scores.get(cid, 0) + (1 / (RRF_K + rank + 1)) * priority * mboost
    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [all_chunks[cid] for cid in top_ids if cid in all_chunks]


async def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    api_key = settings.cohere_api_key.get_secret_value()
    if not api_key or not chunks:
        return sorted(chunks, key=lambda x: x["metadata"].get("rerank_score", 0), reverse=True)[:top_k]
    try:
        co = cohere.AsyncClient(api_key)
        docs = [c.get("text") or "" for c in chunks]
        response = await co.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=docs,
            top_n=top_k,
        )
        await co.close()
        reranked = []
        for result in response.results:
            chunk = chunks[result.index].copy()
            chunk["metadata"]["rerank_score"] = result.relevance_score
            reranked.append(chunk)
        return reranked
    except Exception:
        return chunks[:top_k]


def _row_to_dict(row) -> dict:
    return {
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
            "content_type":     row.content_type or "text",
            "page_number":      row.page_number,
            "figure_number":    row.figure_number,
            "image_path":       row.image_path,
            "table_markdown":   row.table_markdown,
            "table_summary":    row.table_summary,
            "rerank_score":     float(row.similarity),
        }
    }


async def retrieve(
    query: str,
    top_k: int = 5,
    db=None,
    content_type: str | None = None,
    use_hyde: bool = False,
    paper_id: str | None = None,
) -> list[dict]:
    if db is None:
        raise ValueError("db session required")

    modality_boosts = _detect_modality_boost(query)

    if use_hyde:
        from src.retrieval.hyde import expand_query_with_hyde
        _, hyde_docs, _ = expand_query_with_hyde(query, n_docs=2)
        # FIX: embed ONLY the hyde docs (true HyDE), never mix in the raw
        # query embedding. Averaging query+hyde dragged the vector toward
        # generic territory, causing wrong-paper retrieval on specific
        # technical questions (verified: pulled survey papers over the
        # specific paper they were surveying).
        embeddings = embed_texts(hyde_docs)
        query_emb = np.mean(embeddings, axis=0).tolist()
    else:
        import asyncio, functools
        loop = asyncio.get_event_loop()
        query_emb = await loop.run_in_executor(
            None, functools.partial(lambda q: embed_texts([q])[0].tolist(), query)
        )

    vec_results = await pgvector_search(query_emb, top_k=top_k * 3, db=db, content_type=content_type, paper_id=paper_id)
    fts_results = await postgres_fts_search(query, top_k=top_k * 3, db=db, content_type=content_type, paper_id=paper_id)

    fused = rrf_fuse(vec_results, fts_results, top_k=top_k, modality_boosts=modality_boosts)
    return await rerank(query, fused, top_k=top_k)
