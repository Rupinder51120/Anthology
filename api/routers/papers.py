from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from pathlib import Path
from sqlalchemy import text
from api.core.database import get_db
from api.schemas.schemas import PaperOut, PaperListResponse
from api.services.paper_service import PaperService
from api.core.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["Papers"])
paper_service = PaperService()
settings = get_settings()


@router.get("/papers", response_model=PaperListResponse)
async def list_papers(db: AsyncSession = Depends(get_db)):
    papers = await paper_service.get_all_papers(db)
    return PaperListResponse(papers=papers, total=len(papers))


@router.get("/papers/{paper_id}", response_model=PaperOut)
async def get_paper(paper_id: UUID, db: AsyncSession = Depends(get_db)):
    paper = await paper_service.get_paper_by_id(paper_id, db)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.post("/papers/sync")
async def sync_papers(db: AsyncSession = Depends(get_db)):
    synced = await paper_service.sync_registry_to_db(db)
    return {"success": True, "synced": synced}


@router.post("/vectors/search")
async def vector_search(
    query: str,
    top_k: int = 5,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(text("""
        SELECT chunk_id, source, title, text,
               ts_rank_cd(to_tsvector('english', text),
                          plainto_tsquery('english', :q)) as rank
        FROM chunks
        WHERE to_tsvector('english', text) @@ plainto_tsquery('english', :q)
        ORDER BY rank DESC
        LIMIT :k
    """), {"q": query, "k": top_k})
    rows = result.fetchall()
    results = [{"chunk_id": r.chunk_id, "source": r.source,
                "title": r.title, "text": r.text,
                "similarity": float(r.rank)} for r in rows]
    return {"query": query, "results": results, "total": len(results)}
