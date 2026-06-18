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
from langfuse import Langfuse

from src.generation.generator import generate_answer, format_citations
from src.generation.memory import ConversationMemory

CACHE_TTL = 3600        # 1 hour
MAX_SESSIONS = 1000     # FIX: evict oldest sessions beyond this cap


def _get_langfuse():
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
    )


_sessions: dict[str, ConversationMemory] = {}
_session_access: dict[str, float] = {}  # FIX: track last access for LRU eviction


def _get_memory(session_id: str) -> ConversationMemory:
    # FIX: evict LRU sessions when cap reached
    if session_id not in _sessions and len(_sessions) >= MAX_SESSIONS:
        oldest = min(_session_access, key=_session_access.get)
        _sessions.pop(oldest, None)
        _session_access.pop(oldest, None)

    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(session_id=session_id)

    _session_access[session_id] = time.time()
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
            try:
                cached = await redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    data["latency_ms"] = round((time.time() - start) * 1000, 2)
                    data["query_id"]   = str(uuid.uuid4())
                    await redis.aclose()  # FIX: always close after use
                    return QueryResponse(**data)
            except Exception:
                pass  # redis failure is non-fatal
            finally:
                pass

        session_id   = getattr(request, "session_id", "default") or "default"
        memory       = _get_memory(session_id)
        chat_history = memory.get()
        lf = _get_langfuse()

        trace = None
        try:
            trace = lf.trace(name="anthology-query", input={"question": request.question})
        except Exception:
            trace = None

        t0 = time.time()
        chunks = await retrieve(
            query=request.question,
            top_k=request.top_k,
            db=db,
            paper_id=request.paper_id,
        )
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
        if trace:
            try:
                trace.update(
                    output={"answer": result.get("answer", "")[:200]},
                    metadata={"latency_ms": latency_ms},
                )
                lf.flush()
            except Exception:
                pass

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
        # FIX: remove explicit commit here — get_db() commits on exit to avoid double-commit
        # await db.commit()  ← removed

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

        if redis:
            try:
                cache_data = {
                    "question":      request.question,
                    "answer":        result.get("answer", ""),
                    "citations":     [c.model_dump() for c in citations],
                    "chunks_used":   result.get("chunks_used", 0),
                    "response_type": result.get("response_type", "explanation"),
                    "tokens_used":   result.get("tokens_used", 0),
                }
                await redis.setex(cache_key, CACHE_TTL, json.dumps(cache_data))
            except Exception:
                pass
            finally:
                await redis.aclose()

        return response

    async def get_query_count(self, db: AsyncSession) -> int:
        result = await db.execute(select(func.count(Query.id)))
        return result.scalar() or 0
