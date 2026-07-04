from sqlalchemy.ext.asyncio import AsyncSession

from src.retrieval.retriever import retrieve as retrieve_chunks


class RetrievalService:
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        db: AsyncSession | None = None,
        content_type: str | None = None,
        use_hyde: bool = False,
        paper_id: str | None = None,
    ) -> list[dict]:
        return await retrieve_chunks(
            query=query,
            top_k=top_k,
            db=db,
            content_type=content_type,
            use_hyde=use_hyde,
            paper_id=paper_id,
        )
