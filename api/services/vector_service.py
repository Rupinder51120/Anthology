import json
import numpy as np
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from api.models.tables import Chunk
from api.core.config import get_settings

settings = get_settings()


class VectorService:

    async def sync_chunks_to_db(self, db: AsyncSession) -> dict:
        """Load chunks + embeddings from files into PostgreSQL with pgvector."""

        chunks_path = Path(settings.chunks_path)
        embeddings_path = Path("indexes/chunk_embeddings.npy")

        if not chunks_path.exists():
            return {"success": False, "error": "chunks_metadata.json not found — run build_index.py first"}
        if not embeddings_path.exists():
            return {"success": False, "error": "chunk_embeddings.npy not found — run build_index.py first"}

        # Load chunks metadata
        with open(chunks_path) as f:
            chunks = json.load(f)

        # Load embeddings
        embeddings = np.load(str(embeddings_path))

        if len(chunks) != len(embeddings):
            return {
                "success": False,
                "error": f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            }

        # Clear existing chunks
        await db.execute(delete(Chunk))
        await db.commit()

        # Insert in batches
        batch_size = 100
        total = len(chunks)
        inserted = 0

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            db_chunks = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                meta = chunk["metadata"]
                # Clean null bytes from text (PDF parsing artifact)
                clean_text = chunk["text"].replace("\x00", "").replace("\0", "")

                db_chunk = Chunk(
                    chunk_id=meta.get("chunk_id", f"{meta['source']}::{i}").replace("\x00", ""),
                    source=meta.get("source", ""),
                    title=meta.get("title", ""),
                    authors=meta.get("authors", "") if isinstance(
                        meta.get("authors"), str) else ", ".join(meta.get("authors", [])),
                    year=int(meta.get("year")) if meta.get("year") else None,
                    section=meta.get("section", ""),
                    section_priority=meta.get("section_priority", 0.5),
                    chunk_index=meta.get("chunk_index", 0),
                    chunk_type=meta.get("chunk_type", "general"),
                    text=clean_text,
                    char_count=meta.get("char_count", len(chunk["text"])),
                    word_count=meta.get("word_count", len(chunk["text"].split())),
                    embedding=embedding.tolist(),
                )
                db_chunks.append(db_chunk)

            db.add_all(db_chunks)
            await db.commit()
            inserted += len(db_chunks)
            print(f"  Inserted {inserted}/{total} chunks")

        return {
            "success": True,
            "total_chunks": total,
            "inserted": inserted,
            "embedding_dim": embeddings.shape[1],
        }

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        db: AsyncSession = None,
    ) -> list[dict]:
        """Search chunks by vector similarity using pgvector."""
        from sqlalchemy import text

        query_vec = f"[{','.join(str(x) for x in query_embedding)}]"

        result = await db.execute(
            text(f"""
                SELECT
                    chunk_id, source, title, authors, year,
                    section, section_priority, chunk_type, text,
                    1 - (embedding <=> '{query_vec}'::vector) as similarity
                FROM chunks
                ORDER BY embedding <=> '{query_vec}'::vector
                LIMIT {top_k}
            """)
        )

        rows = result.fetchall()
        return [
            {
                "chunk_id": row.chunk_id,
                "source": row.source,
                "title": row.title,
                "authors": row.authors,
                "year": row.year,
                "section": row.section,
                "section_priority": row.section_priority,
                "chunk_type": row.chunk_type,
                "text": row.text,
                "similarity": float(row.similarity),
            }
            for row in rows
        ]
