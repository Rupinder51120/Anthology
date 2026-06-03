import numpy as np
import json
import arxiv
from pathlib import Path
from src.embedder import load_embeddings, embed_texts


def load_paper_meta(path: str = "indexes/paper_meta.json") -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as f:
        return json.load(f)


def recommend_local(
    query_filename: str,
    top_k: int = 3,
    min_similarity: float = 0.5
) -> list[dict]:
    """Find similar papers in local folder by embedding similarity."""
    embeddings = load_embeddings("indexes/paper_embeddings.npy")
    paper_meta = load_paper_meta()

    source_idx = next(
        (i for i, m in enumerate(paper_meta) if m["filename"] == query_filename),
        None
    )

    if source_idx is None:
        return []

    source_vec = embeddings[source_idx]
    scores     = embeddings @ source_vec
    scores[source_idx] = -1  # exclude self

    top_indices = np.argsort(scores)[::-1][:top_k * 2]

    results = []
    for idx in top_indices:
        sim = float(scores[idx])
        if sim < min_similarity:
            continue
        meta = paper_meta[idx]
        results.append({
            "title":      meta["title"],
            "authors":    meta["authors"],
            "year":       meta["year"],
            "filename":   meta["filename"],
            "similarity": round(sim, 3)
        })
        if len(results) >= top_k:
            break

    return results


def recommend_by_query(
    query: str,
    top_k: int = 3,
    min_similarity: float = 0.4
) -> list[dict]:
    """Find papers in local folder most relevant to a query string."""
    embeddings  = load_embeddings("indexes/paper_embeddings.npy")
    paper_meta  = load_paper_meta()
    query_emb   = embed_texts([query])[0]
    scores      = embeddings @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_k * 2]

    results = []
    for idx in top_indices:
        sim = float(scores[idx])
        if sim < min_similarity:
            continue
        meta = paper_meta[idx]
        results.append({
            "title":      meta["title"],
            "authors":    meta["authors"],
            "year":       meta["year"],
            "filename":   meta["filename"],
            "similarity": round(sim, 3)
        })
        if len(results) >= top_k:
            break

    return results


def recommend_arxiv(topic: str, top_k: int = 3) -> list[dict]:
    """Query Arxiv API for related papers not in local folder."""
    try:
        client = arxiv.Client(
            page_size=top_k + 5,
            delay_seconds=1.0,
            num_retries=2
        )
        search = arxiv.Search(
            query=topic,
            max_results=top_k + 5,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title":    paper.title,
                "authors":  ", ".join(a.name for a in paper.authors[:3]),
                "year":     str(paper.published.year),
                "url":      paper.entry_id,
                "abstract": paper.summary[:250] + "..."
            })
            if len(results) >= top_k:
                break

        return results

    except Exception as e:
        print(f"Arxiv API error: {e}")
        return []