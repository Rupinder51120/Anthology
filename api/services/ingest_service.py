"""
Handles single-paper ingestion for the upload endpoint.
"""
import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

BATCH_SIZE = 32  # NEW: bounds peak memory for embeddings/insert payloads


def _sync_ingest(pdf_path: str) -> dict:
    """Run all CPU-bound ingestion synchronously (parse, caption, chunk only — no embedding)."""
    from src.ingestion.parser import parse_pdf
    from src.ingestion.chunker import chunk_parsed_blocks
    from src.ingestion.ingest import extract_metadata_from_pdf
    from src.ingestion.table_summarizer import summarize_table
    from src.ingestion.figure_captioner import caption_figure
    # REMOVED: from src.retrieval.embedder import embed_chunks  ← moved to batch loop in ingest_single_paper

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

    # NOTE: embeddings deliberately NOT computed here anymore.
    return {"chunks": chunks, "meta": meta, "filename": filename}


def _embed_batch_sync(batch: list[dict]):
    """Thin sync wrapper so embed_chunks can be dispatched to the executor per-batch."""
    from src.retrieval.embedder import embed_chunks
    return embed_chunks(batch)


async def ingest_single_paper(pdf_path: str, db: AsyncSession) -> dict:
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _sync_ingest, pdf_path)
    if "error" in data:
        return data

    chunks   = data["chunks"]
    meta     = data["meta"]
    filename = data["filename"]

    inserted = 0
    fig_count = 0
    tbl_count = 0

    async with db.begin():
        await db.execute(
            text("DELETE FROM chunks WHERE source = :source"),
            {"source": filename},
        )

        # NEW: process in fixed-size batches instead of all at once
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]

            # CPU-bound embedding for THIS batch only, off the event loop
            embeddings_batch = await loop.run_in_executor(None, _embed_batch_sync, batch)

            for chunk, emb in zip(batch, embeddings_batch):
                m = chunk["metadata"]
                if m.get("content_type") == "figure":
                    fig_count += 1
                elif m.get("content_type") == "table":
                    tbl_count += 1

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

            # NEW: drop references so the batch + its embeddings can be GC'd
            # before the next iteration allocates the next batch's embeddings.
            del embeddings_batch, batch
    # db.begin() commits here on clean exit; rolls back on any exception above.

    return {
        "filename": filename,
        "title": meta.get("title", ""),
        "chunks": inserted,
        "figures": fig_count,
        "tables": tbl_count,
    }