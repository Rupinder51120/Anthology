"""
Multimodal retriever: pgvector + FTS + RRF + modality filtering.
"""
from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

RRF_K = 60

MODALITY_BOOST = {
    "figure":   ["figure", "diagram", "architecture", "shows", "image", "chart", "plot", "visualization"],
    "table":    ["table", "results", "comparison", "benchmark", "scores", "metrics", "performance"],
    "equation": ["equation", "formula", "loss", "objective", "proof"],
}


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
    db: AsyncSession,
    content_type: str | None = None,
) -> list[dict]:
    query_vec = "[" + ",".join(str(x) for x in query_embedding) + "]"
    type_filter = f"AND content_type = '{content_type}'" if content_type else ""
    result = await db.execute(text(f"""
        SELECT id, paper_id, content_type, content, page_number,
               section_title, figure_number, image_path,
               table_markdown, table_summary,
               1 - (embedding <=> '{query_vec}'::vector) as similarity
        FROM chunks
        WHERE embedding IS NOT NULL {type_filter}
        ORDER BY embedding <=> '{query_vec}'::vector
        LIMIT {top_k}
    """))
    return [_row_to_dict(r) for r in result.fetchall()]


async def fts_search(
    query: str,
    top_k: int,
    db: AsyncSession,
    content_type: str | None = None,
) -> list[dict]:
    type_filter = f"AND content_type = '{content_type}'" if content_type else ""
    sql = (
        "SELECT id, paper_id, content_type, content, page_number,"
        " section_title, figure_number, image_path,"
        " table_markdown, table_summary,"
        " ts_rank(to_tsvector('english', content),"
        "     websearch_to_tsquery('english', :q)) as similarity"
        " FROM chunks"
        " WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', :q)"
        " {}" 
        " ORDER BY similarity DESC LIMIT {}"
    ).format(type_filter, top_k)
    try:
        result = await db.execute(text(sql), {"q": query})
        return [_row_to_dict(r) for r in result.fetchall()]
    except Exception as e:
        print(f"FTS ERROR: {e}")
        return []


def rrf_fuse(
    result_lists: list[list[dict]],
    top_k: int,
    boosts: dict[str, float] | None = None,
) -> list[dict]:
    all_chunks = {}
    scores: dict[str, float] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            cid = chunk["id"]
            all_chunks[cid] = chunk
            boost = (boosts or {}).get(chunk["content_type"], 1.0)
            scores[cid] = scores.get(cid, 0) + boost / (RRF_K + rank + 1)

    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [all_chunks[cid] for cid in top_ids]


async def retrieve(
    query: str,
    top_k: int = 5,
    db: AsyncSession = None,
    content_type_filter: str | None = None,
    use_vlm_embeddings: bool = True,
) -> list[dict]:
    if db is None:
        raise ValueError("db required")

    boosts = _detect_modality_boost(query)

    if use_vlm_embeddings:
        from multimodal.ingestion.embedder import embed_texts
        query_emb = embed_texts([query])[0].tolist()
        vec_results = await pgvector_search(query_emb, top_k * 3, db, content_type_filter)
    else:
        vec_results = []

    fts_results = await fts_search(query, top_k * 3, db, content_type_filter)
    fused = rrf_fuse([vec_results, fts_results], top_k, boosts)
    return fused


def _row_to_dict(row) -> dict:
    return {
        "id":             str(row.id),
        "paper_id":       str(row.paper_id),
        "content_type":   row.content_type,
        "content":        row.content,
        "page_number":    row.page_number,
        "section_title":  row.section_title,
        "figure_number":  row.figure_number,
        "image_path":     row.image_path,
        "table_markdown": row.table_markdown,
        "table_summary":  row.table_summary,
        "score":          float(row.similarity),
    }
