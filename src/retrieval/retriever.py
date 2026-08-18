"""
src/retrieval/retriever.py
SPECTER2 + pgvector + FTS + modality-boosted RRF + Cohere rerank

GATE 2 FIXES vs prior version:
1. `strategy` param added so each retrieval stage (sparse / dense /
   dense+hyde / hybrid+rrf / hybrid+rerank / dense+rerank) can be
   benchmarked in isolation, not just the bundled production path.
2. BM25 keyword terms extracted from HyDE docs were generated but never
   used — postgres_fts_search() ran on the raw query only. Now the
   extracted terms are appended to the FTS query (bounded, deduped)
   whenever HyDE runs.
3. rrf_fuse() previously truncated to `top_k` BEFORE rerank(), so Cohere
   could only reorder within a pool it never had a chance to fix.
   rrf_fuse() now accepts a separate `fuse_k` (wider than top_k) and the
   final top_k cut happens only inside rerank().
4. expand_query_with_hyde() call is now awaited via
   expand_query_with_hyde_async() (see hyde.py) instead of running
   synchronously inside the async request path.
"""

import os
import numpy as np
import cohere
from src.retrieval.embedder import embed_texts
from api.core.models import COHERE_RERANK_MODEL, CROSS_ENCODER_MODEL
from api.core.config import get_settings

settings = get_settings()
RRF_K = 60
RERANK_POOL_MULTIPLIER = 2  # GATE 2 FIX: candidates surviving into rerank = top_k * this

VALID_STRATEGIES = {
    "sparse",         # FTS only
    "dense",          # pgvector only, raw query embedding
    "dense_hyde",      # pgvector only, HyDE-mean query embedding
    "hybrid_rrf",      # dense + sparse fused via RRF, no rerank
    "hybrid_rerank",   # dense + sparse fused via RRF, then Cohere rerank (production default)
    "dense_rerank",    # pgvector only, then Cohere rerank (no FTS/RRF)
}

_cross_encoder = None  # retained for future local-rerank fallback; not currently invoked

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
            section, section_priority, chunk_index, chunk_type, content_type,
            text, char_count, word_count,
            page_number, figure_number, image_path,
            table_markdown, table_summary, is_enriched,
            1 - (embedding <=> CAST(:vec AS vector)) as similarity
        FROM chunks
        {filters}
        ORDER BY embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """.format(filters=filters))
    result = await db.execute(sql, params)
    return [_row_to_dict(r) for r in result.fetchall()]


def _build_fts_query(query: str, bm25_terms: list[str] | None, max_terms: int = 12) -> str:
    """
    GATE 2 FIX: fold HyDE-derived BM25 terms into the FTS query so
    postgres_fts_search actually benefits from HyDE's keyword extraction
    instead of discarding it. Bounded to max_terms to avoid an
    unbounded/noisy tsquery on long HyDE outputs.
    """
    if not bm25_terms:
        return query
    extra = " ".join(bm25_terms[:max_terms])
    return f"{query} {extra}"


async def postgres_fts_search(
    query: str,
    top_k: int,
    db,
    content_type: str | None = None,
    paper_id: str | None = None,
    bm25_terms: list[str] | None = None,
) -> list[dict]:
    from sqlalchemy import text

    fts_query = _build_fts_query(query, bm25_terms)

    # Use OR semantics for natural-language questions. plainto_tsquery()
    # joins all terms with AND, which made paraphrased benchmark questions
    # return zero sparse candidates.
    filters = """WHERE to_tsvector('english', text) @@
        websearch_to_tsquery(
            'english',
            regexp_replace(:query, '\\s+', ' OR ', 'g')
        )"""
    params: dict = {"query": fts_query, "k": top_k}
    if content_type:
        filters += " AND content_type = :ct"
        params["ct"] = content_type
    if paper_id:
        filters += " AND paper_id = :pid"
        params["pid"] = paper_id

    sql = text("""
        SELECT
            chunk_id, source, title, authors, year,
            section, section_priority, chunk_index, chunk_type, content_type,
            text, char_count, word_count,
            page_number, figure_number, image_path,
            table_markdown, table_summary, is_enriched,
            ts_rank_cd(
                to_tsvector('english', text),
                websearch_to_tsquery(
                    'english',
                    regexp_replace(:query, '\\s+', ' OR ', 'g')
                )
            ) as similarity
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
    fuse_k: int,
    modality_boosts: dict = None,
) -> list[dict]:
    """
    GATE 2 FIX: takes `fuse_k` (candidate pool size) instead of the final
    `top_k`. Callers pass a wider fuse_k (e.g. top_k * RERANK_POOL_MULTIPLIER)
    when the output feeds into rerank(), so Cohere has real candidates to
    choose from instead of re-sorting an already-truncated top_k list.
    """
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
    top_ids = sorted(scores, key=scores.get, reverse=True)[:fuse_k]
    return [all_chunks[cid] for cid in top_ids if cid in all_chunks]


async def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    api_key = settings.cohere_api_key.get_secret_value()
    if not api_key or not chunks:
        return sorted(chunks, key=lambda x: x["metadata"].get("rerank_score", 0), reverse=True)[:top_k]

    import asyncio
    retries = 3
    backoff = [1, 2, 4]

    for attempt in range(retries + 1):
        try:
            co = cohere.AsyncClient(api_key)
            docs = [c.get("text") or "" for c in chunks]
            response = await co.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=docs,
                top_n=top_k,
            )
            reranked = []
            for result in response.results:
                chunk = chunks[result.index].copy()
                chunk["metadata"]["rerank_score"] = result.relevance_score
                reranked.append(chunk)
            return reranked
        except Exception as e:
            if attempt < retries:
                wait = backoff[attempt]
                print(f"Cohere rerank 429/error: {e}. Retrying in {wait}s... ({attempt+1}/{retries})")
                await asyncio.sleep(wait)
            else:
                print(f"Cohere rerank failed after {retries} retries: {type(e).__name__}: {e}")
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
            "chunk_index":      getattr(row, "chunk_index", 0) or 0,
            "chunk_type":       row.chunk_type or "general",
            "content_type":     row.content_type or "text",
            "char_count":       getattr(row, "char_count", 0) or 0,
            "word_count":       getattr(row, "word_count", 0) or 0,
            "page_number":      row.page_number,
            "figure_number":    row.figure_number,
            "image_path":       row.image_path,
            "table_markdown":   row.table_markdown,
            "table_summary":    row.table_summary,
            "is_enriched":      bool(getattr(row, "is_enriched", False)),
            "rerank_score":     float(row.similarity),
        }
    }


async def _embed_query_raw(query: str) -> list[float]:
    import asyncio, functools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, functools.partial(lambda q: embed_texts([q])[0].tolist(), query)
    )


async def _embed_query_hyde(query: str) -> tuple[list[float], list[str]]:
    """Returns (hyde-mean embedding, bm25_terms)."""
    from src.retrieval.hyde import expand_query_with_hyde_async
    _, hyde_docs, bm25_terms = await expand_query_with_hyde_async(query, n_docs=2)
    embeddings = embed_texts(hyde_docs)
    query_emb = np.mean(embeddings, axis=0).tolist()
    return query_emb, bm25_terms


async def retrieve(
    query: str,
    top_k: int = 5,
    db=None,
    content_type: str | None = None,
    use_hyde: bool = False,
    paper_id: str | None = None,
    strategy: str = "hybrid_rerank",
) -> list[dict]:
    """
    strategy:
        sparse         — FTS only
        dense          — pgvector only, raw query embedding
        dense_hyde     — pgvector only, HyDE-mean query embedding
        hybrid_rrf     — dense + sparse fused via RRF, no rerank
        hybrid_rerank  — dense + sparse fused via RRF, then Cohere rerank (default)
        dense_rerank   — pgvector only, then Cohere rerank
    """
    if db is None:
        raise ValueError("db session required")
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {sorted(VALID_STRATEGIES)}")

    modality_boosts = _detect_modality_boost(query)
    bm25_terms: list[str] = []

    # --- resolve query embedding + bm25 terms per strategy -------------
    needs_dense = strategy in {"dense", "dense_hyde", "hybrid_rrf", "hybrid_rerank", "dense_rerank"}
    needs_sparse = strategy in {"sparse", "hybrid_rrf", "hybrid_rerank"}
    hyde_requested = use_hyde or strategy == "dense_hyde"

    query_emb = None
    if needs_dense:
        if hyde_requested:
            query_emb, bm25_terms = await _embed_query_hyde(query)
        else:
            query_emb = await _embed_query_raw(query)

    # --- fetch candidates ------------------------------------------------
    fetch_k = top_k * 3  # wide net for fusion/rerank stages
    vec_results, fts_results = [], []

    if needs_dense:
        vec_results = await pgvector_search(
            query_emb, top_k=fetch_k, db=db, content_type=content_type, paper_id=paper_id
        )
    if needs_sparse:
        fts_results = await postgres_fts_search(
            query, top_k=fetch_k, db=db, content_type=content_type, paper_id=paper_id,
            bm25_terms=bm25_terms,
        )

    # --- combine per strategy --------------------------------------------
    if strategy == "sparse":
        return fts_results[:top_k]

    if strategy in {"dense", "dense_hyde"}:
        return vec_results[:top_k]

    if strategy == "hybrid_rrf":
        return rrf_fuse(vec_results, fts_results, fuse_k=top_k, modality_boosts=modality_boosts)

    if strategy == "dense_rerank":
        return await rerank(query, vec_results[: top_k * RERANK_POOL_MULTIPLIER], top_k=top_k)

    if strategy == "hybrid_rerank":
        fused = rrf_fuse(
            vec_results, fts_results,
            fuse_k=top_k * RERANK_POOL_MULTIPLIER,   # GATE 2 FIX: wider pool survives into rerank
            modality_boosts=modality_boosts,
        )
        return await rerank(query, fused, top_k=top_k)

    raise AssertionError("unreachable")  # VALID_STRATEGIES guard above should prevent this