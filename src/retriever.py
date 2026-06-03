"""
src/retriever.py — Fixed HyDE integration

Key changes vs original:
1. embed ONLY hyde_docs (not query+hyde concat) — true HyDE behaviour
2. Average embeddings of N hypothetical docs → stable centroid
3. BM25 query augmented with HyDE keywords (not just original query)
4. Kept all prior fixes (#3-#8) intact
"""

import numpy as np
import json
import re
import time
from sentence_transformers import CrossEncoder
from src.embedder import embed_texts
from src.indexer import load_faiss_index, load_bm25_index

_cross_encoder = None
_chunks_cache  = None
_faiss_cache   = None
_bm25_cache    = None

USE_RERANKER = False


# ─── loaders with caching ─────────────────────────────────────

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("Loading cross-encoder...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def load_chunks(path: str = "indexes/chunks_metadata.json") -> list[dict]:
    global _chunks_cache
    if _chunks_cache is None:
        with open(path) as f:
            _chunks_cache = json.load(f)
    return _chunks_cache


def get_faiss_index():
    global _faiss_cache
    if _faiss_cache is None:
        _faiss_cache = load_faiss_index()
    return _faiss_cache


def get_bm25_index():
    global _bm25_cache
    if _bm25_cache is None:
        _bm25_cache = load_bm25_index()
    return _bm25_cache


# ─── query intent detection ───────────────────────────────────

def detect_query_intent(query: str) -> str:
    q = query.lower()
    math_words    = ['equation', 'formula', 'loss function', 'derivative',
                     'gradient', 'proof', 'theorem', 'calculate', 'compute']
    compare_words = ['compare', 'difference', 'versus', 'vs', 'better',
                     'advantage', 'disadvantage', 'trade-off']
    concept_words = ['how', 'why', 'what is', 'explain', 'understand',
                     'intuition', 'meaning', 'define']
    if any(w in q for w in math_words):
        return "math"
    if any(w in q for w in compare_words):
        return "comparison"
    if any(w in q for w in concept_words):
        return "concept"
    return "search"


# ─── search functions ─────────────────────────────────────────

def faiss_search(query_embedding: np.ndarray, top_k: int = 25) -> list[int]:
    index  = get_faiss_index()
    query  = query_embedding.astype("float32")
    scores, indices = index.search(query, top_k)
    if hasattr(indices, 'ndim') and indices.ndim == 2:
        indices = indices[0]
    return [int(i) for i in indices if int(i) >= 0]


def bm25_search(query: str, chunks: list[dict], top_k: int = 25,
                extra_terms: list[str] = None) -> list[int]:
    bm25   = get_bm25_index()
    tokens = re.findall(r'\b[a-z][a-z0-9]{1,}\b', query.lower())
    # FIX: augment with HyDE keyword terms for richer BM25 signal
    if extra_terms:
        tokens = tokens + extra_terms
    if not tokens:
        return []
    scores      = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [int(i) for i in top_indices if scores[i] > 0]


def reciprocal_rank_fusion(
    faiss_ids:    list[int],
    bm25_ids:     list[int],
    k:            int   = 60,
    faiss_weight: float = 1.0,
    bm25_weight:  float = 1.0,
) -> list[int]:
    scores = {}
    for rank, doc_id in enumerate(faiss_ids):
        scores[doc_id] = scores.get(doc_id, 0) + faiss_weight / (rank + k)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (rank + k)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    seen   = set()
    unique = []
    for c in candidates:
        key = c["text"][:100].strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def boost_by_section_priority(candidates: list[dict]) -> list[dict]:
    return sorted(
        candidates,
        key=lambda c: c["metadata"].get("section_priority", 0.5),
        reverse=True
    )


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []
    cross_encoder = get_cross_encoder()
    pairs         = [(query, c["text"][:512]) for c in candidates]
    scores        = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in ranked[:top_k]:
        doc["metadata"]["rerank_score"] = round(float(score), 4)
        results.append(doc)
    return results


# ─── HyDE embedding helper ────────────────────────────────────

def _hyde_embedding(hyde_docs: list[str]) -> np.ndarray:
    """
    FIX #1 (core HyDE fix): embed each hypothetical doc SEPARATELY,
    then average the embeddings.  This gives a stable centroid in the
    document embedding space — geometrically much closer to real answer
    chunks than the original query embedding.

    The original code did embed(query + "\n\n" + hyde_doc) which is wrong:
    it moves the vector only partway toward the document space.
    """
    embeddings = embed_texts(hyde_docs)          # shape: (N, dim)
    avg = np.mean(embeddings, axis=0)            # shape: (dim,)
    # L2-normalise so cosine similarity works correctly
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg


# ─── main retrieve ────────────────────────────────────────────

def retrieve(
    query:        str,
    top_k:        int   = 5,
    faiss_weight: float = 1.0,
    bm25_weight:  float = 1.0,
    use_hyde:     bool  = False,
) -> list[dict]:

    chunks = load_chunks()
    intent = detect_query_intent(query)   # always on original query

    # ── intent-based weight adjustment ──
    if intent == "math":
        bm25_weight  = 1.4
    elif intent == "concept":
        faiss_weight = 1.4
    elif intent == "comparison":
        faiss_weight = 1.1
        bm25_weight  = 1.1

    bm25_extra_terms: list[str] = []

    # ── HyDE: embed ONLY the hypothetical docs, not the query ──
    t = time.time()
    if use_hyde:
        try:
            from src.hyde import expand_query_with_hyde
            _, hyde_docs, bm25_extra_terms = expand_query_with_hyde(query, n_docs=3)
            # FIX: average embedding of N hyde docs (true HyDE)
            query_embedding = _hyde_embedding(hyde_docs)
            print(f"  HyDE: {len(hyde_docs)} docs generated, "
                  f"{len(bm25_extra_terms)} BM25 terms")
        except Exception as e:
            print(f"HyDE failed, falling back to query embedding: {e}")
            query_embedding = embed_texts([query])[0]
    else:
        query_embedding = embed_texts([query])[0]
    print(f"  embed: {int((time.time()-t)*1000)}ms")

    t = time.time()
    faiss_ids = faiss_search(query_embedding, top_k=25)
    print(f"  faiss: {int((time.time()-t)*1000)}ms")

    t = time.time()
    # FIX: pass BM25 keywords extracted from HyDE docs for richer keyword recall
    bm25_ids = bm25_search(query, chunks, top_k=25, extra_terms=bm25_extra_terms)
    print(f"  bm25:  {int((time.time()-t)*1000)}ms")

    fused_ids = reciprocal_rank_fusion(
        faiss_ids, bm25_ids,
        faiss_weight=faiss_weight,
        bm25_weight=bm25_weight,
    )[:20]

    candidates = [chunks[i] for i in fused_ids if i < len(chunks)]
    candidates = deduplicate_candidates(candidates)

    if USE_RERANKER:
        reranked = rerank(query, candidates, top_k=top_k * 2)
        final    = boost_by_section_priority(reranked)[:top_k]
    else:
        for c in candidates:
            c["metadata"]["rerank_score"] = None
        final = boost_by_section_priority(candidates)[:top_k]

    print(f"Retrieved {len(final)} chunks | intent={intent} | hyde={use_hyde}")
    return final


if __name__ == "__main__":
    results = retrieve("How does the GAN discriminator work?", use_hyde=True)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source:  {r['metadata']['title'][:50]}")
        print(f"Section: {r['metadata']['section']}")
        print(f"Score:   {r['metadata'].get('rerank_score', 'N/A')}")
        print(r["text"][:250])