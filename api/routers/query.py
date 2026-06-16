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
    import os
    from src.retrieval.retriever import retrieve
    from src.generation.generator import _build_messages

    chunks = await retrieve(
        request.question,
        top_k=request.top_k,
        db=db,
        use_hyde=getattr(request, "use_hyde", False),
        paper_id=getattr(request, "paper_id", None),
    )

    _, messages = _build_messages(request.question, chunks)

    async def token_stream():
        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", ""))
            model  = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            stream = await client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=1024, temperature=0.2, stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    import json
                    yield f"data: {json.dumps(delta)}\n\n"
            await client.close()
        except Exception as e:
            yield f"data: Generation failed: {e}\n\n"
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
