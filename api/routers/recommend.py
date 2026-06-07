from fastapi import APIRouter
from api.schemas.schemas import RecommendRequest, RecommendResponse, RecommendationOut

router = APIRouter(prefix="/api/v1", tags=["Recommendations"])


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get paper recommendations — local + arxiv."""
    from src.ui.recommender import recommend_by_query, recommend_arxiv

    local_raw = recommend_by_query(request.query, top_k=request.top_k)
    arxiv_raw = recommend_arxiv(request.query, top_k=request.top_k)

    local = [
        RecommendationOut(
            title=r.get("title", ""),
            authors=r.get("authors", ""),
            year=r.get("year"),
            filename=r.get("filename", ""),
            similarity=r.get("similarity", 0.0),
        )
        for r in local_raw
    ]

    arxiv = [
        RecommendationOut(
            title=r.get("title", ""),
            authors=r.get("authors", ""),
            year=r.get("year"),
            filename="",
            similarity=0.0,
            url=r.get("url"),
        )
        for r in arxiv_raw
    ]

    return RecommendResponse(
        query=request.query,
        local=local,
        arxiv=arxiv,
    )
