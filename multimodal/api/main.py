from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from contextlib import asynccontextmanager
from pathlib import Path
import os, uuid, shutil

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://anthology:anthology@localhost:5433/anthology_multimodal")
FIGURES_DIR  = os.getenv("FIGURES_DIR", "data/figures")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    from multimodal.api.models.tables import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(title="Anthology Multimodal API", version="2.0.0", lifespan=lifespan)

# Serve figure images
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        r = await db.execute(text("SELECT COUNT(*) FROM chunks"))
        count = r.scalar()
        return {"status": "ok", "chunks": count}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/api/v2/papers")
async def list_papers(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT id, filename, title, authors, year,
               chunk_count, figure_count, table_count, indexed, created_at
        FROM papers ORDER BY created_at DESC
    """))
    rows = result.fetchall()
    return {"papers": [dict(r._mapping) for r in rows], "total": len(rows)}


@app.get("/api/v2/papers/{paper_id}/figures")
async def get_figures(paper_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT id, content, figure_number, page_number, image_path, section_title
        FROM chunks
        WHERE paper_id = :pid AND content_type = 'figure'
        ORDER BY page_number
    """), {"pid": paper_id})
    rows = result.fetchall()
    return {"figures": [dict(r._mapping) for r in rows]}


@app.get("/api/v2/papers/{paper_id}/tables")
async def get_tables(paper_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT id, content, figure_number, page_number, table_markdown, table_summary, section_title
        FROM chunks
        WHERE paper_id = :pid AND content_type = 'table'
        ORDER BY page_number
    """), {"pid": paper_id})
    rows = result.fetchall()
    return {"tables": [dict(r._mapping) for r in rows]}


@app.post("/api/v2/query")
async def query(request: dict, db: AsyncSession = Depends(get_db)):
    import time
    from multimodal.retrieval.retriever import retrieve
    from multimodal.api.services.generator import generate_answer

    start = time.time()
    question       = request.get("question", "")
    top_k          = request.get("top_k", 5)
    content_filter = request.get("content_type")

    chunks = await retrieve(
        query=question,
        top_k=top_k,
        db=db,
        content_type_filter=content_filter,
        use_vlm_embeddings=False,  # FTS-only on API; worker handles embeddings
    )

    answer = await generate_answer(question, chunks)
    return {
        "question":    question,
        "answer":      answer,
        "chunks_used": len(chunks),
        "sources":     chunks,
        "latency_ms":  round((time.time() - start) * 1000, 2),
    }


@app.post("/api/v2/ingest")
async def ingest_paper(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "PDF only")

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = upload_dir / file.filename

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Async ingestion — runs in background
    import asyncio
    asyncio.create_task(_run_ingestion(str(pdf_path), file.filename, db))

    return {"status": "ingestion_started", "filename": file.filename}


async def _run_ingestion(pdf_path: str, filename: str, db: AsyncSession):
    from multimodal.ingestion.pipeline import ingest_paper
    try:
        stats = await ingest_paper(
            pdf_path=pdf_path,
            paper_metadata={"filename": filename, "title": filename.replace(".pdf", "").replace("_", " ")},
            session=db,
            use_vlm=True,
        )
        print(f"Ingestion complete: {stats}")
    except Exception as e:
        print(f"Ingestion failed: {e}")
