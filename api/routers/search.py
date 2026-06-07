from fastapi import APIRouter
from api.schemas.schemas import SearchRequest, SearchResponse, SearchResultOut

router = APIRouter(prefix="/api/v1", tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search across all paper chunks."""
    from src.retrieval.retriever import retrieve

    chunks = retrieve(
        request.query,
        top_k=request.top_k,
        use_hyde=request.use_hyde,
    )

    results = [
        SearchResultOut(
            title=c["metadata"].get("title", ""),
            authors=c["metadata"].get("authors", ""),
            year=c["metadata"].get("year"),
            section=c["metadata"].get("section", ""),
            score=c["metadata"].get("rerank_score"),
            text=c["text"][:500],
            filename=c["metadata"].get("source", ""),
        )
        for c in chunks
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
    )
