from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from api.models.tables import Query, Paper
from api.schemas.schemas import StatsResponse


class StatsService:

    async def get_stats(self, db: AsyncSession) -> StatsResponse:
        paper_result = await db.execute(select(func.count(Paper.id)))
        total_papers = paper_result.scalar() or 0

        query_result = await db.execute(select(func.count(Query.id)))
        total_queries = query_result.scalar() or 0

        chunk_result = await db.execute(text("SELECT COUNT(*) FROM chunks"))
        total_chunks = chunk_result.scalar() or 0

        emb_result = await db.execute(text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"))
        vector_chunks = emb_result.scalar() or 0

        return StatsResponse(
            total_papers=total_papers,
            total_chunks=total_chunks,
            vector_chunks=vector_chunks,
            embedding_dim=1024,
            total_queries=total_queries,
        )
