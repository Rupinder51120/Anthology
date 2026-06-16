"""
src/discovery/discovery_service.py
Merge ArXiv + S2 results, deduplicate by title similarity.
"""
import asyncio
from src.discovery.arxiv_client import search_arxiv
from src.discovery.s2_client import search_s2


def _normalize(title: str) -> str:
    return "".join(c.lower() for c in title if c.isalnum())


def _dedupe(papers: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in papers:
        key = _normalize(p.get("title", ""))[:40]
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


async def discover(query: str, max_per_source: int = 8) -> dict:
    arxiv_task = search_arxiv(query, max_results=max_per_source)
    s2_task    = search_s2(query, max_results=max_per_source)

    arxiv_results, s2_results = await asyncio.gather(
        arxiv_task, s2_task, return_exceptions=True
    )

    if isinstance(arxiv_results, Exception):
        arxiv_results = []
    if isinstance(s2_results, Exception):
        s2_results = []

    # sort S2 by citation count
    s2_results = sorted(s2_results, key=lambda x: x.get("citation_count") or 0, reverse=True)

    combined = _dedupe(arxiv_results + s2_results)

    return {
        "query":   query,
        "arxiv":   arxiv_results,
        "s2":      s2_results,
        "combined": combined,
        "total":   len(combined),
    }
