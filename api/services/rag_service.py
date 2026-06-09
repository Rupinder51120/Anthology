import time
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from api.models.tables import Query
from api.schemas.schemas import QueryRequest, QueryResponse, CitationOut

from src.generation.generator import generate_answer, format_citations


class RAGService:

    async def query(
        self,
        request: QueryRequest,
        db: AsyncSession,
    ) -> QueryResponse:
        start = time.time()

        # Retrieve chunks
        import os
        use_pgvector = os.getenv("USE_PGVECTOR", "false").lower() == "true"
        
        if use_pgvector:
            # Cloud mode — pure PostgreSQL FTS, no ML models needed
            from sqlalchemy import text as sql_text
            clean_query = " & ".join(
                w for w in request.question.split() if len(w) > 2
            )
            try:
                result = await db.execute(
                    sql_text("""
                        SELECT chunk_id, source, title, authors, year,
                               section, section_priority, chunk_type, text,
                               ts_rank(to_tsvector('english', text),
                                       to_tsquery('english', :q)) as rank
                        FROM chunks
                        WHERE to_tsvector('english', text) @@ to_tsquery('english', :q)
                        ORDER BY rank DESC
                        LIMIT :k
                    """),
                    {"q": clean_query, "k": request.top_k}
                )
                rows = result.fetchall()
                chunks = [
                    {
                        "text": row.text,
                        "metadata": {
                            "chunk_id":         row.chunk_id,
                            "source":           row.source,
                            "title":            row.title,
                            "authors":          row.authors or "",
                            "year":             row.year,
                            "section":          row.section or "",
                            "section_priority": row.section_priority or 0.5,
                            "chunk_type":       row.chunk_type or "general",
                            "rerank_score":     float(row.rank),
                        }
                    }
                    for row in rows
                ]
            except Exception:
                chunks = []
        else:
            from src.retrieval.retriever import retrieve
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
