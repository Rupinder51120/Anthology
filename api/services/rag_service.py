import time
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from api.models.tables import Query
from api.schemas.schemas import QueryRequest, QueryResponse, CitationOut
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer, format_citations
from src.generation.memory import ConversationMemory

# In-memory sessions — keyed by session_id
_sessions: dict[str, ConversationMemory] = {}

def _get_memory(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(session_id=session_id)
    return _sessions[session_id]


class RAGService:
    async def query(
        self,
        request: QueryRequest,
        db: AsyncSession,
    ) -> QueryResponse:
        start = time.time()

        # Memory
        session_id = getattr(request, "session_id", "default") or "default"
        memory = _get_memory(session_id)
        chat_history = memory.get()

        chunks = await retrieve(
            query=request.question,
            top_k=request.top_k,
            db=db,
        )

        # Pass image_paths for figure chunks
        image_paths = [
            c["metadata"].get("image_path")
            for c in chunks
            if c["metadata"].get("content_type") == "figure"
            and c["metadata"].get("image_path")
        ]

        result = generate_answer(
            query=request.question,
            chunks=chunks,
            chat_history=chat_history,
            image_paths=image_paths if image_paths else None,
        )

        # Update memory
        memory.add("user", request.question)
        memory.add("assistant", result.get("answer", ""))

        latency_ms = round((time.time() - start) * 1000, 2)

        citations = [
            CitationOut(
                title=c.get("title", ""),
                authors=c.get("authors", ""),
                year=c.get("year"),
                section=c.get("section", ""),
                filename=c.get("filename", ""),
                score=c.get("score"),
            )
            for c in result.get("citations", [])
        ]

        query_id = uuid.uuid4()
        db_query = Query(
            id=query_id,
            question=request.question,
            answer=result.get("answer", ""),
            retrieval_mode="pgvector",
            top_k=request.top_k,
            chunks_used=result.get("chunks_used", 0),
            citations=[c.model_dump() for c in citations],
            response_type=result.get("response_type", "explanation"),
            tokens_used=result.get("tokens_used", 0),
            latency_ms=latency_ms,
        )
        db.add(db_query)
        await db.commit()

        return QueryResponse(
            question=request.question,
            answer=result.get("answer", ""),
            citations=citations,
            chunks_used=result.get("chunks_used", 0),
            response_type=result.get("response_type", "explanation"),
            tokens_used=result.get("tokens_used", 0),
            latency_ms=latency_ms,
            query_id=query_id,
        )

    async def get_query_count(self, db: AsyncSession) -> int:
        result = await db.execute(select(func.count(Query.id)))
        return result.scalar() or 0
