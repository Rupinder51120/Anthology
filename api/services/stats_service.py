import json
import numpy as np
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from api.models.tables import Query, Paper
from api.schemas.schemas import StatsResponse
from api.core.config import get_settings

settings = get_settings()


class StatsService:

    async def get_stats(self, db: AsyncSession) -> StatsResponse:
        # Paper count from DB
        paper_result = await db.execute(select(func.count(Paper.id)))
        total_papers = paper_result.scalar() or 0

        # Query count
        query_result = await db.execute(select(func.count(Query.id)))
        total_queries = query_result.scalar() or 0

        # Chunk count from JSON
        total_chunks = 0
        chunks_path = Path(settings.chunks_path)
        if chunks_path.exists():
            with open(chunks_path) as f:
                chunks = json.load(f)
            total_chunks = len(chunks)

        # FAISS info — from DB if pgvector mode
        faiss_vectors = 0
        embedding_dim = 1024
        emb_path = Path("indexes/chunk_embeddings.npy")
        if emb_path.exists():
            emb = np.load(str(emb_path))
            faiss_vectors, embedding_dim = emb.shape
        else:
            # Cloud mode — get count from chunks table
            from sqlalchemy import text as sql_text
            try:
                result = await db.execute(sql_text("SELECT COUNT(*) FROM chunks"))
                faiss_vectors = result.scalar() or 0
            except Exception:
                faiss_vectors = total_chunks

        return StatsResponse(
            total_papers=total_papers,
            total_chunks=total_chunks,
            faiss_vectors=faiss_vectors,
            embedding_dim=embedding_dim,
            total_queries=total_queries,
        )
