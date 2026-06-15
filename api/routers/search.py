from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.core.database import get_db
from api.schemas.schemas import SearchRequest, SearchResponse, SearchResultOut

router = APIRouter(prefix="/api/v1", tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),  # FIX: inject db session
):
    """Semantic search across all paper chunks."""
    from src.retrieval.retriever import retrieve

    # FIX: await async retrieve, pass db
    chunks = await retrieve(
        request.query,
        top_k=request.top_k,
        db=db,
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
