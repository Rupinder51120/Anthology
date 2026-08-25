import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from pathlib import Path
from sqlalchemy import text
from api.core.database import get_db
from api.schemas.schemas import PaperOut, PaperListResponse, AddExternalPaperRequest
from api.services.paper_service import PaperService

router = APIRouter(prefix="/api/v1", tags=["Papers"])
paper_service = PaperService()

UPLOAD_DIR = Path("data/papers")


def _safe_pdf_filename(filename: str | None) -> str:
    """
    Reduce an untrusted upload filename to a basename that cannot escape
    UPLOAD_DIR, regardless of path traversal ('../'), absolute paths, or
    nested separators. Path(...).name strips every directory component
    (relative or absolute) in one step; the remaining checks handle the
    degenerate cases that leaves behind (empty, '.', '..', or a bare
    extension), which would otherwise resolve to UPLOAD_DIR itself or a
    hidden/dot file.
    """
    name = Path((filename or "").strip()).name
    name = name.replace("\\", "_")  # backslash isn't a POSIX separator, but don't allow it to look like one
    stem = name.rsplit(".", 1)[0] if "." in name else name
    if not name or name in (".", "..") or not stem:
        name = f"{uuid.uuid4().hex}.pdf"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name


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

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_pdf_filename(file.filename)
    dest = (UPLOAD_DIR / safe_name).resolve()

    # Defense in depth: _safe_pdf_filename already guarantees a bare
    # basename, so this should never trip, but refuse to write anywhere
    # outside UPLOAD_DIR rather than trust a single layer of sanitization.
    if UPLOAD_DIR.resolve() not in dest.parents:
        raise HTTPException(status_code=400, detail="Invalid filename")

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


@router.post("/papers/add-external")
async def add_external_paper(
    request: AddExternalPaperRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Download a PDF found via external Search (arXiv/OpenAlex) and ingest it
    through the same pipeline as a manual upload. Reuses upload_paper's
    filename-safety and ingestion call unchanged -- this is only a different
    way to get bytes onto disk before ingestion, not a different pipeline.
    """
    import httpx
    from api.services.ingest_service import ingest_single_paper

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_pdf_filename(request.title or "external-paper")
    dest = (UPLOAD_DIR / safe_name).resolve()

    if UPLOAD_DIR.resolve() not in dest.parents:
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(request.pdf_url)
            resp.raise_for_status()
            content = resp.content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not download paper: {e}")

    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=502, detail="The linked file is not a valid PDF")

    if len(content) > 50 * 1024 * 1024:  # 50MB limit, matches manual upload
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    with open(dest, "wb") as f_out:
        f_out.write(content)

    try:
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
