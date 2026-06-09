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


@router.post("/vectors/sync")
async def sync_vectors(db: AsyncSession = Depends(get_db)):
    """Sync chunk embeddings to PostgreSQL pgvector."""
    import traceback
    from api.services.vector_service import VectorService
    try:
        vector_service = VectorService()
        result = await vector_service.sync_chunks_to_db(db)
        return result
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.post("/vectors/search")
async def vector_search(
    query: str,
    top_k: int = 5,
    db: AsyncSession = Depends(get_db),
):
    """Search chunks using PostgreSQL full-text search (cloud) or pgvector (local)."""
    from api.services.vector_service import VectorService
    vector_service = VectorService()

    import os
    if os.getenv("USE_PGVECTOR", "false").lower() == "true":
        # Cloud mode — use PostgreSQL FTS (no embedding model needed)
        from sqlalchemy import text
        clean_query = " & ".join(w for w in query.split() if len(w) > 2)
        result = await db.execute(
            text("""
                SELECT chunk_id, source, title, authors, year,
                       section, chunk_type, text,
                       ts_rank(to_tsvector('english', text),
                               to_tsquery('english', :q)) as rank
                FROM chunks
                WHERE to_tsvector('english', text) @@ to_tsquery('english', :q)
                ORDER BY rank DESC
                LIMIT :k
            """),
            {"q": clean_query, "k": top_k}
        )
        rows = result.fetchall()
        results = [{"chunk_id": r.chunk_id, "source": r.source,
                    "title": r.title, "text": r.text,
                    "similarity": float(r.rank)} for r in rows]
    else:
        # Local mode — use pgvector similarity
        from src.retrieval.embedder import embed_texts
        embedding = embed_texts([query])[0].tolist()
        results = await vector_service.similarity_search(embedding, top_k, db)

    return {"query": query, "results": results, "total": len(results)}
