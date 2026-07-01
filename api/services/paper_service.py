import json
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from api.models.tables import Paper
from api.core.config import get_settings

settings = get_settings()


class PaperService:

    async def sync_registry_to_db(self, db: AsyncSession) -> int:
        """Sync download_registry.json into PostgreSQL papers table."""
        registry_path = Path(settings.registry_path)
        if not registry_path.exists():
            return 0

        with open(registry_path) as f:
            registry = json.load(f)

        # Load chunk counts
        chunk_counts = {}
        chunks_path = Path(settings.chunks_path)
        if chunks_path.exists():
            with open(chunks_path) as f:
                chunks = json.load(f)
            for chunk in chunks:
                src = chunk["metadata"]["source"]
                chunk_counts[src] = chunk_counts.get(src, 0) + 1

        synced = 0
        for arxiv_id, meta in registry.items():
            # Check if already exists
            result = await db.execute(
                select(Paper).where(Paper.arxiv_id == arxiv_id)
            )
            existing = result.scalar_one_or_none()

            filename = meta.get("filename", "")
            chunk_count = chunk_counts.get(filename, 0)

            if existing:
                existing.chunk_count = chunk_count
                existing.indexed = chunk_count > 0
            else:
                paper = Paper(
                    arxiv_id=arxiv_id,
                    filename=filename,
                    title=meta.get("title", "Unknown"),
                    authors=", ".join(meta.get("authors", [])) if isinstance(
                        meta.get("authors"), list) else meta.get("authors", ""),
                    abstract=meta.get("abstract", ""),
                    year=meta.get("year"),
                    topic=meta.get("topic", ""),
                    url=meta.get("url", ""),
                    doi=meta.get("doi"),
                    chunk_count=chunk_count,
                    indexed=chunk_count > 0,
                )
                db.add(paper)
                synced += 1

        await db.commit()
        return synced

    async def get_all_papers(self, db: AsyncSession) -> list[Paper]:
        result = await db.execute(
            select(Paper).order_by(Paper.year.desc())
        )
        return result.scalars().all()

    async def get_paper_by_id(self, paper_id, db: AsyncSession) -> Paper | None:
        result = await db.execute(
            select(Paper).where(Paper.id == paper_id)
        )
        return result.scalar_one_or_none()

    async def get_total_count(self, db: AsyncSession) -> int:
        result = await db.execute(select(Paper))
        return len(result.scalars().all())
