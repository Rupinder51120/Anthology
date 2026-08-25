from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.core.database import get_db
from api.schemas.schemas import SearchRequest, SearchResponse, SearchResultOut
from api.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/v1", tags=["Search"])
retrieval_service = RetrievalService()


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),  # FIX: inject db session
):
    """Semantic search across all paper chunks."""

    # FIX: await async retrieve, pass db
    chunks = await retrieval_service.retrieve(
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
            paper_id=c["metadata"].get("paper_id"),
        )
        for c in chunks
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
    )
