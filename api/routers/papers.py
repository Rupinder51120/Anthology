from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from pathlib import Path
from sqlalchemy import text
from api.core.database import get_db
from api.schemas.schemas import PaperOut, PaperListResponse
from api.services.paper_service import PaperService

router = APIRouter(prefix="/api/v1", tags=["Papers"])
paper_service = PaperService()


@router.post("/papers/upload")
async def upload_paper(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF and ingest it into the system."""
    from api.services.ingest_service import ingest_single_paper

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    dest = Path("data/papers") / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    with open(dest, "wb") as f_out:
        f_out.write(content)

    try:
        # FIX: directly await — no nested asyncio.run() inside executor
        result = await ingest_single_paper(str(dest), db)
        return {"success": True, **result}
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


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
    # FIX: parameterized query — no f-string SQL injection
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
    results = [
        {
            "chunk_id": r.chunk_id,
            "source": r.source,
            "title": r.title,
            "text": r.text,
            "similarity": float(r.rank),
        }
        for r in rows
    ]
    return {"query": query, "results": results, "total": len(results)}
