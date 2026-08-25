import time
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from api.core.database import get_db
from api.schemas.schemas import QueryRequest, QueryResponse
from api.services.rag_service import RAGService, _get_langfuse
from api.services.retrieval_service import RetrievalService
from api.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Query"])
rag_service = RAGService()
retrieval_service = RetrievalService()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    return await rag_service.query(request, db)


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    import json
    from src.generation.generator import stream_answer, collect_image_paths

    async def token_stream():
        start = time.time()
        lf = _get_langfuse()
        trace = None
        try:
            trace = lf.trace(name="anthology-query-stream", input={"question": request.question})
        except Exception:
            trace = None

        # ── Phase 1: retrieval ──────────────────────────────────
        yield f"data: {json.dumps({'type': 'status', 'text': 'Searching your papers...'})}\n\n"

        t0 = time.time()
        try:
            chunks = await retrieval_service.retrieve(
                request.question,
                top_k=request.top_k,
                db=db,
                use_hyde=getattr(request, "use_hyde", False),
                paper_id=getattr(request, "paper_id", None),
            )
        except Exception as e:
            # A retrieval failure (e.g. a malformed paper_id) must not abort
            # the HTTP response mid-stream -- that leaves the client's fetch
            # reader hanging forever with no [DONE], no error, nothing.
            logger.error("Retrieval call failed: %s", e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'text': 'The system encountered an error while searching your papers.'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        if trace:
            try:
                trace.span(
                    name="retrieve",
                    input={"query": request.question},
                    output={"chunks": len(chunks)},
                    metadata={"latency_ms": round((time.time() - t0) * 1000, 2)},
                )
            except Exception:
                pass

        yield f"data: {json.dumps({'type': 'status', 'text': f'Reranking {len(chunks)} chunks...'})}\n\n"

        image_paths = collect_image_paths(chunks)
        if image_paths and settings.use_groq:
            yield f"data: {json.dumps({'type': 'status', 'text': f'Found {len(image_paths)} relevant figure(s) -- using vision-capable generation...'})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'text': 'Generating answer...'})}\n\n"

        # ── Phase 2: generation (same provider config as /query) ─
        full_answer = ""
        async for event in stream_answer(request.question, chunks, image_paths=image_paths):
            if event["type"] == "token":
                full_answer += event["text"]
            yield f"data: {json.dumps(event)}\n\n"

        if trace:
            try:
                trace.update(
                    output={"answer": full_answer[:200]},
                    metadata={"latency_ms": round((time.time() - start) * 1000, 2)},
                )
                lf.flush()
            except Exception:
                pass

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
