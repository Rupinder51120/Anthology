"""
Handles single-paper ingestion for the upload endpoint.
"""
import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


def _sync_ingest(pdf_path: str) -> dict:
    """Run all CPU-bound ingestion synchronously."""
    from src.ingestion.parser import parse_pdf
    from src.ingestion.chunker import chunk_parsed_blocks
    from src.ingestion.ingest import extract_metadata_from_pdf
    from src.ingestion.table_summarizer import summarize_table
    from src.ingestion.figure_captioner import caption_figure
    from src.retrieval.embedder import embed_chunks

    path = Path(pdf_path)
    filename = path.name
    blocks = parse_pdf(str(path))
    first_page = blocks[0].content if blocks else ""
    meta = extract_metadata_from_pdf(first_page, filename)

    for block in blocks:
        if block.content_type == "figure" and block.image_path:
            try:
                result = caption_figure(block.image_path, meta.get("title", ""), block.figure_number or "Figure")
                block.content = result["caption"]
                if result["table_data"]:
                    block.table_markdown = result["table_data"]
                    block.content_type = "table"
            except Exception as e:
                print(f"Caption failed: {e}")
        elif block.content_type == "table" and block.table_markdown:
            try:
                summary = summarize_table(block.table_markdown, meta.get("title", ""))
                if summary:
                    block.content = f"{block.content}\n\nSummary: {summary}"
            except Exception as e:
                print(f"Table summary failed: {e}")

    chunks = chunk_parsed_blocks(blocks, meta)
    if not chunks:
        return {"chunks": [], "meta": meta, "filename": filename, "error": "No chunks extracted"}

    embeddings = embed_chunks(chunks)
    return {"chunks": chunks, "embeddings": embeddings, "meta": meta, "filename": filename}


async def ingest_single_paper(pdf_path: str, db: AsyncSession) -> dict:
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _sync_ingest, pdf_path)
    if "error" in data:
        return data

    chunks     = data["chunks"]
    embeddings = data["embeddings"]
    meta       = data["meta"]
    filename   = data["filename"]

    
    inserted = 0
    async with db.begin():
        await db.execute(
            text("DELETE FROM chunks WHERE source = :source"),
            {"source": filename},
        )

        for chunk, emb in zip(chunks, embeddings):
            m = chunk["metadata"]
            vec = "[" + ",".join(str(x) for x in emb.tolist()) + "]"
            await db.execute(text("""
                INSERT INTO chunks (
                    chunk_id, source, title, authors, year, section,
                    section_priority, chunk_index, chunk_type, content_type,
                    text, char_count, word_count, page_number, figure_number,
                    image_path, table_markdown, table_summary, embedding
                ) VALUES (
                    :chunk_id, :source, :title, :authors, :year, :section,
                    :section_priority, :chunk_index, :chunk_type, :content_type,
                    :text, :char_count, :word_count, :page_number, :figure_number,
                    :image_path, :table_markdown, :table_summary, :embedding::vector
                ) ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
            """), {
                "chunk_id": m.get("chunk_id", ""),
                "source": m.get("source", filename),
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "year": meta.get("year"),
                "section": m.get("section", ""),
                "section_priority": m.get("section_priority", 0.5),
                "chunk_index": m.get("chunk_index", 0),
                "chunk_type": m.get("chunk_type", "general"),
                "content_type": m.get("content_type", "text"),
                "text": chunk["text"],
                "char_count": m.get("char_count", 0),
                "word_count": m.get("word_count", 0),
                "page_number": m.get("page_number"),
                "figure_number": m.get("figure_number"),
                "image_path": m.get("image_path"),
                "table_markdown": m.get("table_markdown"),
                "table_summary": m.get("table_summary"),
                "embedding": vec,
            })
            inserted += 1
    # db.begin() commits here on clean exit; rolls back on any exception above.
    return {
        "filename": filename,
        "title": meta.get("title", ""),
        "chunks": inserted,
        "figures": sum(1 for c in chunks if c["metadata"].get("content_type") == "figure"),
        "tables": sum(1 for c in chunks if c["metadata"].get("content_type") == "table"),
    }