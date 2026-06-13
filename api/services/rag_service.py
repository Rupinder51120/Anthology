import time
import uuid
import json
import hashlib
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from api.models.tables import Query
from api.schemas.schemas import QueryRequest, QueryResponse, CitationOut
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer, format_citations
from src.generation.memory import ConversationMemory

CACHE_TTL = 3600  # 1 hour

_sessions: dict[str, ConversationMemory] = {}


def _get_memory(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(session_id=session_id)
    return _sessions[session_id]


def _cache_key(question: str, top_k: int) -> str:
    h = hashlib.md5(f"{question.strip().lower()}:{top_k}".encode()).hexdigest()
    return f"anthology:query:{h}"


async def _get_redis():
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        await r.ping()
        return r
    except Exception:
        return None


class RAGService:
    async def query(
        self,
        request: QueryRequest,
        db: AsyncSession,
    ) -> QueryResponse:
        start = time.time()

        # Redis cache check
        redis = await _get_redis()
        cache_key = _cache_key(request.question, request.top_k)
        if redis:
            cached = await redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                data["latency_ms"] = round((time.time() - start) * 1000, 2)
                data["query_id"]   = str(uuid.uuid4())
                return QueryResponse(**data)

        # Memory
        session_id   = getattr(request, "session_id", "default") or "default"
        memory       = _get_memory(session_id)
        chat_history = memory.get()

        chunks = await retrieve(
            query=request.question,
            top_k=request.top_k,
            db=db,
        )

        image_paths = [
            c["metadata"].get("image_path")
            for c in chunks
            if c["metadata"].get("content_type") == "figure"
            and c["metadata"].get("image_path")
        ]

        result = await generate_answer(
            query=request.question,
            chunks=chunks,
            chat_history=chat_history,
            image_paths=image_paths if image_paths else None,
        )

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

        response = QueryResponse(
            question=request.question,
            answer=result.get("answer", ""),
            citations=citations,
            chunks_used=result.get("chunks_used", 0),
            response_type=result.get("response_type", "explanation"),
            tokens_used=result.get("tokens_used", 0),
            latency_ms=latency_ms,
            query_id=query_id,
        )

        # Store in Redis
        if redis:
            cache_data = {
                "question":      request.question,
                "answer":        result.get("answer", ""),
                "citations":     [c.model_dump() for c in citations],
                "chunks_used":   result.get("chunks_used", 0),
                "response_type": result.get("response_type", "explanation"),
                "tokens_used":   result.get("tokens_used", 0),
            }
            await redis.setex(cache_key, CACHE_TTL, json.dumps(cache_data))
            await redis.aclose()

        return response

    async def get_query_count(self, db: AsyncSession) -> int:
        result = await db.execute(select(func.count(Query.id)))
        return result.scalar() or 0
