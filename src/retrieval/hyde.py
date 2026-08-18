"""
src/retrieval/hyde.py — HyDE implementation

GATE 1 FIX vs prior version:
- _generate_one_doc() uses a blocking `requests.post()`. Calling
  expand_query_with_hyde() directly from an async route (as retriever.py
  did) blocked the FastAPI event loop for the full Ollama generation time
  (2 docs x ~350 tokens each) on every HyDE query. Under concurrent
  requests this stalls unrelated requests on the same worker.
- Added expand_query_with_hyde_async(), which runs the existing sync
  function in a threadpool executor. retriever.py now calls this instead
  of the sync version. The sync version is kept for callers (scripts,
  offline eval) that don't need loop safety.
"""

import re
import asyncio
import functools
import numpy as np
import os
import requests
from api.core.models import OLLAMA_CHAT_MODEL

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"

HYDE_PROMPT = """You are a research scientist writing a section of a peer-reviewed paper.
Answer the following research question with a dense, technical paragraph (6-8 sentences).
Use precise terminology. Include specific mechanisms, metrics, or formulas where relevant.
Do NOT say "In this paper" or "We propose". Write as if explaining to an expert.

Research question: {query}

Technical answer:"""


def _generate_one_doc(query: str, temperature: float = 0.55) -> str:
    """Generate a single hypothetical document. BLOCKING — call via executor from async code."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model":  OLLAMA_CHAT_MODEL,
            "prompt": HYDE_PROMPT.format(query=query),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 350},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def extract_bm25_keywords(hyde_doc: str) -> list[str]:
    """
    Pull technical keywords from the HyDE doc for BM25/FTS augmentation.
    Targets multi-char domain tokens unlikely to appear in the original
    short query.
    """
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
        and not t.isupper()
    ]
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:40]


def expand_query_with_hyde(
    query: str,
    n_docs: int = 3,
) -> tuple[str, list[str], list[str]]:
    """
    SYNC / BLOCKING. Prefer expand_query_with_hyde_async from async routes.

    Returns:
        query        — original query (unchanged)
        hyde_docs    — list of N hypothetical document strings
        bm25_terms   — deduplicated keyword terms for FTS augmentation
    """
    hyde_docs = []
    temps = [0.5, 0.6, 0.55][:n_docs]

    for i in range(n_docs):
        try:
            doc = _generate_one_doc(query, temperature=temps[i % len(temps)])
            if doc and len(doc.split()) >= 30:
                hyde_docs.append(doc)
        except Exception as e:
            print(f"HyDE doc {i+1}/{n_docs} failed: {e}")

    if not hyde_docs:
        print("All HyDE generations failed — falling back to original query.")
        return query, [query], []

    all_keywords: list[str] = []
    seen_kw: set[str] = set()
    for doc in hyde_docs:
        for kw in extract_bm25_keywords(doc):
            if kw not in seen_kw:
                seen_kw.add(kw)
                all_keywords.append(kw)

    return query, hyde_docs, all_keywords[:60]


async def expand_query_with_hyde_async(
    query: str,
    n_docs: int = 3,
) -> tuple[str, list[str], list[str]]:
    """
    GATE 1 FIX: non-blocking entry point. Runs the sync HyDE pipeline in
    a threadpool so the event loop stays free for other requests while
    Ollama generates.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, functools.partial(expand_query_with_hyde, query, n_docs)
    )