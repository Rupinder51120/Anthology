"""
src/discovery/s2_client.py
Fetch papers from Semantic Scholar API.
"""
import httpx

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,authors,year,externalIds,openAccessPdf,citationCount"


async def search_s2(query: str, max_results: int = 8) -> list[dict]:
    params = {
        "query":  query,
        "limit":  max_results,
        "fields": FIELDS,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(S2_API, params=params)
        if resp.status_code == 429:
            return []
        resp.raise_for_status()

    data    = resp.json()
    papers  = data.get("data", [])
    results = []

    for p in papers:
        try:
            authors = [a["name"] for a in (p.get("authors") or [])]
            pdf_url = None
            oap     = p.get("openAccessPdf")
            if oap:
                pdf_url = oap.get("url")
            arxiv_id = (p.get("externalIds") or {}).get("ArXiv")
            results.append({
                "s2_id":          p.get("paperId", ""),
                "arxiv_id":       arxiv_id,
                "title":          p.get("title", ""),
                "abstract":       (p.get("abstract") or "")[:400],
                "authors":        ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                "year":           p.get("year"),
                "url":            f"https://www.semanticscholar.org/paper/{p.get('paperId','')}",
                "pdf_url":        pdf_url or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None),
                "citation_count": p.get("citationCount", 0),
                "source":         "semantic_scholar",
            })
        except Exception:
            continue
    return results
