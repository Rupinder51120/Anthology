from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["Discovery"])


class DiscoveryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=300)
    max_per_source: int = Field(default=8, ge=1, le=20)


@router.post("/discover")
async def discover_papers(request: DiscoveryRequest):
    """Search ArXiv + Semantic Scholar for papers matching query."""
    from src.discovery.discovery_service import discover
    return await discover(request.query, max_per_source=request.max_per_source)
