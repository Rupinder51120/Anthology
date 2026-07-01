import re
"""
src/hyde.py — Fixed HyDE implementation

Key fixes vs original:
1. Multi-hypothesis: generate N docs, average their embeddings (stable centroid)
2. Embed ONLY the HyDE doc, never the query — this is what HyDE actually does
3. Extract BM25 keyword boost terms from HyDE doc (technical nouns/verbs)
4. Temperature 0.55: coherent but diverse (0.7 was too random, 0.3 too flat)
5. More tokens (350) for richer vocabulary coverage
"""

import re
import numpy as np
import requests
from api.core.models import OLLAMA_CHAT_MODEL

OLLAMA_URL = "http://localhost:11434/api/generate"

HYDE_PROMPT = """You are a research scientist writing a section of a peer-reviewed paper.
Answer the following research question with a dense, technical paragraph (6-8 sentences).
Use precise terminology. Include specific mechanisms, metrics, or formulas where relevant.
Do NOT say "In this paper" or "We propose". Write as if explaining to an expert.

Research question: {query}

Technical answer:"""


def _generate_one_doc(query: str, temperature: float = 0.55) -> str:
    """Generate a single hypothetical document."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model":  OLLAMA_CHAT_MODEL,
            "prompt": HYDE_PROMPT.format(query=query),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 350},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def extract_bm25_keywords(hyde_doc: str) -> list[str]:
    """
    Pull technical keywords from the HyDE doc for BM25 augmentation.
    Targets: multi-word noun phrases and domain-specific tokens that are
    unlikely to appear in the original short query.
    """
    # keep tokens that are ≥4 chars, not pure stopwords
    stopwords = {
        "this", "that", "with", "from", "have", "been", "they",
        "their", "which", "when", "where", "such", "these", "those",
        "used", "using", "also", "more", "than", "into", "over",
        "each", "both", "through", "about", "after", "between",
    }
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]{3,}\b', hyde_doc)
    keywords = [
        t.lower() for t in tokens
        if t.lower() not in stopwords
        and not t.isupper()   # skip ALL-CAPS acronyms (too noisy)
    ]
    # deduplicate, keep order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:40]   # cap at 40 terms


def expand_query_with_hyde(
    query: str,
    n_docs: int = 3,
) -> tuple[str, list[str], list[str]]:
    """
    Returns:
        query        — original query (unchanged)
        hyde_docs    — list of N hypothetical document strings
        bm25_terms   — deduplicated keyword terms for BM25 augmentation
    """
    hyde_docs = []
    # vary temperature slightly across runs for diversity
    temps = [0.5, 0.6, 0.55][:n_docs]

    for i in range(n_docs):
        try:
            doc = _generate_one_doc(query, temperature=temps[i % len(temps)])
            if doc and len(doc.split()) >= 30:   # sanity check: non-empty
                hyde_docs.append(doc)
        except Exception as e:
            print(f"HyDE doc {i+1}/{n_docs} failed: {e}")

    if not hyde_docs:
        print("All HyDE generations failed — falling back to original query.")
        return query, [query], []

    # BM25 keywords: union across all docs, then deduplicate
    all_keywords: list[str] = []
    seen_kw: set[str] = set()
    for doc in hyde_docs:
        for kw in extract_bm25_keywords(doc):
            if kw not in seen_kw:
                seen_kw.add(kw)
                all_keywords.append(kw)

    return query, hyde_docs, all_keywords[:60]