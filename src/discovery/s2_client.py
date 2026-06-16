"""
src/discovery/s2_client.py
OpenAlex API — replaces Semantic Scholar (no API key needed)
250M+ papers, full citation graph
"""
import httpx

OPENALEX_API = "https://api.openalex.org/works"


async def search_s2(query: str, max_results: int = 8) -> list[dict]:
    """Search OpenAlex — kept as search_s2 for interface compatibility."""
    params = {
        "search":     query,
        "per-page":   max_results,
        "sort":       "relevance_score:desc",
        "select":     "id,title,abstract_inverted_index,authorships,publication_year,cited_by_count,primary_location,ids,concepts",
        "mailto":     "anthology@research.dev",  # polite pool = faster responses
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(OPENALEX_API, params=params)
        if resp.status_code != 200:
            return []
        data = resp.json()

    results = []
    for work in data.get("results", []):
        try:
            # reconstruct abstract from inverted index
            abstract = ""
            inv = work.get("abstract_inverted_index") or {}
            if inv:
                words = {}
                for word, positions in inv.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words[k] for k in sorted(words))[:400]

            authors_raw = work.get("authorships", [])
            authors = ", ".join(
                a["author"]["display_name"]
                for a in authors_raw[:3]
                if a.get("author", {}).get("display_name")
            )
            if len(authors_raw) > 3:
                authors += " et al."

            ids       = work.get("ids", {})
            arxiv_id  = ids.get("arxiv", "").replace("https://arxiv.org/abs/", "") if ids.get("arxiv") else None
            doi       = ids.get("doi", "")
            url       = work.get("primary_location", {}).get("landing_page_url") or \
                        (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"https://doi.org/{doi}" if doi else "")
            pdf_url   = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None

            results.append({
                "s2_id":          work.get("id", ""),
                "arxiv_id":       arxiv_id,
                "title":          work.get("title", ""),
                "abstract":       abstract,
                "authors":        authors,
                "year":           work.get("publication_year"),
                "url":            url,
                "pdf_url":        pdf_url,
                "citation_count": work.get("cited_by_count", 0),
                "source":         "semantic_scholar",  # kept for UI badge compat
            })
        except Exception:
            continue
    return results
