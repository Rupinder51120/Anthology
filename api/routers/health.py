import requests
from fastapi import APIRouter
from api.schemas.schemas import HealthResponse
from api.core.config import get_settings
from pathlib import Path

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    # Check Ollama
    ollama_ok = False
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_ok = resp.status_code == 200
    except Exception:
        pass

    # Check index
    index_ok = Path(settings.chunks_path).exists()

    return HealthResponse(
        status="ok" if (ollama_ok and index_ok) else "degraded",
        version=settings.app_version,
        ollama=ollama_ok,
        index=index_ok,
    )
