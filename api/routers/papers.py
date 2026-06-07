from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from pathlib import Path
from api.core.database import get_db
from api.schemas.schemas import PaperOut, PaperListResponse
from api.services.paper_service import PaperService
from api.core.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["Papers"])
paper_service = PaperService()
settings = get_settings()


@router.get("/papers", response_model=PaperListResponse)
async def list_papers(db: AsyncSession = Depends(get_db)):
    """List all indexed papers."""
    papers = await paper_service.get_all_papers(db)
    return PaperListResponse(
        papers=papers,
        total=len(papers),
    )


@router.get("/papers/{paper_id}", response_model=PaperOut)
async def get_paper(
    paper_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get paper detail by ID."""
    paper = await paper_service.get_paper_by_id(paper_id, db)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.post("/papers/upload")
async def upload_paper(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a new PDF and add it to the index."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    papers_dir = Path(settings.papers_dir)
    papers_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = papers_dir / file.filename

    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        from src.ingestion.index_manager import add_paper
        add_paper(pdf_path)
        await paper_service.sync_registry_to_db(db)
        return {"success": True, "message": f"Added {file.filename} to collection"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/papers/sync")
async def sync_papers(db: AsyncSession = Depends(get_db)):
    """Sync registry to PostgreSQL."""
    synced = await paper_service.sync_registry_to_db(db)
    return {"success": True, "synced": synced}
