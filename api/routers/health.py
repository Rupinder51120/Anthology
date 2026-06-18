import os
import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from api.core.database import get_db
from api.schemas.schemas import HealthResponse
from api.core.config import get_settings
router = APIRouter(tags=["Health"])
settings = get_settings()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

async def _check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False

@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    # Check pgvector index
    try:
        result = await db.execute(text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"))
        index_ok = (result.scalar() or 0) > 0
    except Exception:
        index_ok = False

    ollama_ok = await _check_ollama()

    return HealthResponse(
        status="ok" if index_ok else "degraded",
        version=settings.app_version,
        ollama=ollama_ok,
        index=index_ok,
    )