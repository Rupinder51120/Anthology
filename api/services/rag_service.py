import time
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from api.models.tables import Query
from api.schemas.schemas import QueryRequest, QueryResponse, CitationOut
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer, format_citations


class RAGService:

    async def query(
        self,
        request: QueryRequest,
        db: AsyncSession,
    ) -> QueryResponse:
        start = time.time()

        # Retrieve chunks
        chunks = retrieve(
            request.question,
            top_k=request.top_k,
            use_hyde=request.use_hyde,
        )

        # Generate answer
        result = generate_answer(request.question, chunks)

        latency_ms = round((time.time() - start) * 1000, 2)

        # Build citations
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

        # Save to DB
        query_id = uuid.uuid4()
        db_query = Query(
            id=query_id,
            question=request.question,
            answer=result.get("answer", ""),
            retrieval_mode=request.retrieval_mode,
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
