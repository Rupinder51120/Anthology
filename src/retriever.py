import numpy as np
import json
import re
from pathlib import Path
from sentence_transformers import CrossEncoder
from src.embedder import embed_texts
from src.indexer import load_faiss_index, load_bm25_index

_cross_encoder = None
_chunks_cache  = None
_faiss_cache   = None
_bm25_cache    = None


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

    math_words = ['equation', 'formula', 'loss function', 'derivative',
                  'gradient', 'proof', 'theorem', 'calculate', 'compute']
    if any(w in q for w in math_words):
        return "math"

    compare_words = ['compare', 'difference', 'versus', 'vs', 'better',
                     'advantage', 'disadvantage', 'trade-off']
    if any(w in q for w in compare_words):
        return "comparison"

    concept_words = ['how', 'why', 'what is', 'explain', 'understand',
                     'intuition', 'meaning', 'define']
    if any(w in q for w in concept_words):
        return "concept"

    return "search"


# ─── search functions ─────────────────────────────────────────

def faiss_search(query_embedding: np.ndarray, top_k: int = 20) -> list[int]:
    """Works with both NumpyIndex and real FAISS index."""
    index  = get_faiss_index()
    query  = query_embedding.astype("float32")

    scores, indices = index.search(query, top_k)

    # NumpyIndex returns 1D indices directly
    # real FAISS returns 2D — handle both
    if hasattr(indices, 'ndim') and indices.ndim == 2:
        indices = indices[0]

    return [int(i) for i in indices if int(i) >= 0]


def bm25_search(query: str, chunks: list[dict], top_k: int = 20) -> list[int]:
    bm25   = get_bm25_index()
    tokens = re.findall(r'\b[a-z][a-z0-9]{1,}\b', query.lower())
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
    bm25_weight:  float = 1.0
) -> list[int]:
    scores = {}
    for rank, doc_id in enumerate(faiss_ids):
        scores[doc_id] = scores.get(doc_id, 0) + faiss_weight / (rank + k)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (rank + k)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


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

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    for score, doc in ranked[:top_k]:
        doc["metadata"]["rerank_score"] = round(float(score), 4)
        results.append(doc)

    return results


# ─── main retrieve ────────────────────────────────────────────

def retrieve(
    query:        str,
    top_k:        int   = 5,
    faiss_weight: float = 1.0,
    bm25_weight:  float = 1.0
) -> list[dict]:

    chunks = load_chunks()
    intent = detect_query_intent(query)

    if intent == "math":
        bm25_weight  = 1.4
    elif intent == "concept":
        faiss_weight = 1.4
    elif intent == "comparison":
        faiss_weight = 1.1
        bm25_weight  = 1.1

    query_embedding = embed_texts([query])[0]

    faiss_ids = faiss_search(query_embedding, top_k=20)
    bm25_ids  = bm25_search(query, chunks, top_k=20)

    fused_ids  = reciprocal_rank_fusion(
        faiss_ids, bm25_ids,
        faiss_weight=faiss_weight,
        bm25_weight=bm25_weight
    )[:20]

    candidates = [chunks[i] for i in fused_ids if i < len(chunks)]
    candidates = boost_by_section_priority(candidates)
    final      = rerank(query, candidates, top_k=top_k)

    print(f"Retrieved {len(final)} chunks | intent={intent}")
    return final


if __name__ == "__main__":
    results = retrieve("How does the GAN discriminator work?")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source:  {r['metadata']['title'][:50]}")
        print(f"Section: {r['metadata']['section']}")
        print(f"Score:   {r['metadata'].get('rerank_score', 'N/A')}")
        print(r["text"][:250])
        