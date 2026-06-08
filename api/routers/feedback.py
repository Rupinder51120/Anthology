from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from api.core.database import get_db
from api.models.tables import Feedback, Query
from api.schemas.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/v1", tags=["Feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit rating and comment for a query response."""
    # Verify query exists
    result = await db.execute(
        select(Query).where(Query.id == request.query_id)
    )
    query = result.scalar_one_or_none()
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    # Check if feedback already exists
    existing = await db.execute(
        select(Feedback).where(Feedback.query_id == request.query_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Feedback already submitted")

    feedback = Feedback(
        query_id=request.query_id,
        rating=request.rating,
        comment=request.comment,
    )
    db.add(feedback)
    await db.commit()

    return FeedbackResponse(
        success=True,
        message=f"Feedback submitted — rating {request.rating}/5"
    )


@router.get("/feedback/{query_id}")
async def get_feedback(
    query_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get feedback for a specific query."""
    result = await db.execute(
        select(Feedback).where(Feedback.query_id == query_id)
    )
    feedback = result.scalar_one_or_none()
    if not feedback:
        raise HTTPException(status_code=404, detail="No feedback found")
    return {
        "query_id": str(feedback.query_id),
        "rating": feedback.rating,
        "comment": feedback.comment,
        "created_at": feedback.created_at,
    }
