from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from api.core.database import get_db
from api.schemas.schemas import QueryRequest, QueryResponse
from api.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["Query"])
rag_service = RAGService()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Ask a question — get answer + citations from the paper corpus."""
    return await rag_service.query(request, db)


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Stream answer tokens via SSE as they arrive from Groq."""
    import asyncio
    from src.retrieval.retriever import retrieve
    from src.generation.generator import generate_answer_streaming

    chunks = await retrieve(
        request.question,
        top_k=request.top_k,
        db=db,
        use_hyde=getattr(request, "use_hyde", False),
    )

    async def token_stream():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _produce():
            try:
                for token in generate_answer_streaming(request.question, chunks):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        loop.run_in_executor(None, _produce)

        while True:
            token = await queue.get()
            if token is None:
                break
            # SSE format — React EventSource reads data: lines
            yield f"data: {token}\n\n"

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
