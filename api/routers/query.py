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
async def query_stream(request: QueryRequest):
    """Stream answer tokens one by one."""
    from src.retrieval.retriever import retrieve
    from src.generation.generator import generate_answer_streaming

    chunks = retrieve(
        request.question,
        top_k=request.top_k,
        use_hyde=request.use_hyde,
    )

    def token_generator():
        for token in generate_answer_streaming(request.question, chunks):
            yield token

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
    )
