from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


class VectorService:

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        db: AsyncSession,
    ) -> list[dict]:
        query_vec = "[" + ",".join(str(x) for x in query_embedding) + "]"
        result = await db.execute(text(f"""
            SELECT chunk_id, source, title, text,
                   1 - (embedding <=> '{query_vec}'::vector) as similarity
            FROM chunks
            ORDER BY embedding <=> '{query_vec}'::vector
            LIMIT {top_k}
        """))
        return [
            {
                "chunk_id":   r.chunk_id,
                "source":     r.source,
                "title":      r.title,
                "text":       r.text,
                "similarity": float(r.similarity),
            }
            for r in result.fetchall()
        ]
