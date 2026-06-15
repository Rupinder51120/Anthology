from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


class VectorService:
    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        db: AsyncSession,
        content_type: str | None = None,
    ) -> list[dict]:
        # FIX: SQL injection removed — embedding passed as bound param via ::vector cast
        query_vec = "[" + ",".join(str(x) for x in query_embedding) + "]"

        if content_type:
            sql = text("""
                SELECT
                    chunk_id, source, title, authors, year,
                    section, section_priority, chunk_type, content_type,
                    text, char_count, word_count,
                    page_number, figure_number, image_path,
                    table_markdown, table_summary,
                    1 - (embedding <=> :vec::vector) as similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND content_type = :ct
                ORDER BY embedding <=> :vec::vector
                LIMIT :k
            """)
            result = await db.execute(sql, {"vec": query_vec, "ct": content_type, "k": top_k})
        else:
            sql = text("""
                SELECT
                    chunk_id, source, title, authors, year,
                    section, section_priority, chunk_type, content_type,
                    text, char_count, word_count,
                    page_number, figure_number, image_path,
                    table_markdown, table_summary,
                    1 - (embedding <=> :vec::vector) as similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :vec::vector
                LIMIT :k
            """)
            result = await db.execute(sql, {"vec": query_vec, "k": top_k})

        return [
            {
                "text": r.text,
                "metadata": {
                    "chunk_id":         r.chunk_id,
                    "source":           r.source,
                    "title":            r.title,
                    "authors":          r.authors or "",
                    "year":             r.year,
                    "section":          r.section or "",
                    "section_priority": r.section_priority or 0.5,
                    "chunk_type":       r.chunk_type or "general",
                    "content_type":     r.content_type or "text",
                    "page_number":      r.page_number,
                    "figure_number":    r.figure_number,
                    "image_path":       r.image_path,
                    "table_markdown":   r.table_markdown,
                    "table_summary":    r.table_summary,
                    "rerank_score":     float(r.similarity),
                }
            }
            for r in result.fetchall()
        ]
