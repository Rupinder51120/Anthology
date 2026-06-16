"""
src/discovery/arxiv_client.py
Fetch papers from ArXiv API by query.
"""
import httpx
import xml.etree.ElementTree as ET
from typing import Optional


ARXIV_API = "https://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"


async def search_arxiv(query: str, max_results: int = 8) -> list[dict]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(ARXIV_API, params=params)
        resp.raise_for_status()

    root = ET.fromstring(resp.text)
    results = []
    for entry in root.findall(f"{NS}entry"):
        try:
            arxiv_id = entry.find(f"{NS}id").text.strip().split("/abs/")[-1]
            title    = entry.find(f"{NS}title").text.strip().replace("\n", " ")
            summary  = entry.find(f"{NS}summary").text.strip().replace("\n", " ")
            authors  = [a.find(f"{NS}name").text for a in entry.findall(f"{NS}author")]
            published = entry.find(f"{NS}published").text[:4]  # year
            url      = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url  = f"https://arxiv.org/pdf/{arxiv_id}"

            results.append({
                "arxiv_id":  arxiv_id,
                "title":     title,
                "abstract":  summary[:400],
                "authors":   ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                "year":      int(published),
                "url":       url,
                "pdf_url":   pdf_url,
                "source":    "arxiv",
            })
        except Exception:
            continue
    return results
