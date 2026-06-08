from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.core.database import get_db
from api.schemas.schemas import StatsResponse
from api.services.stats_service import StatsService

router = APIRouter(prefix="/api/v1", tags=["Stats"])
stats_service = StatsService()


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get index and system statistics."""
    return await stats_service.get_stats(db)
