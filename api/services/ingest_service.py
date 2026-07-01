from __future__ import annotations
import time
import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

BATCH_SIZE = 32   # bounds peak memory for embeddings + insert payloads


# ── Sync workers (run in executor) ────────────────────────────────────────────

def _sync_parse_and_enrich(pdf_path: str) -> dict:
    """
    CPU-bound stage: parse → metadata → enrich → chunk.
    No embedding — kept separate so batching can bound memory.

    Returns dict with keys: chunks, meta, filename
    On failure: dict with additional key "error" (str).
    """
    from src.ingestion.parser          import parse_pdf_document
    from src.ingestion.chunker         import chunk_parsed_blocks
    from src.ingestion.metadata_resolver import resolve_metadata
    from src.ingestion.table_summarizer  import summarize_table
    from src.ingestion.figure_captioner  import caption_figure
    func_start = time.perf_counter()
    path     = Path(pdf_path)
    filename = path.name

    # 1. Parse
    parsed_doc = parse_pdf_document(str(path))
    if not parsed_doc.blocks:
        logger.info("_sync_parse_and_enrich took %.2fs", time.perf_counter() - func_start)
        return {"error": "Docling extracted 0 blocks — PDF may be image-only or corrupt",
                "chunks": [], "meta": {}, "filename": filename}

    blocks = parsed_doc.blocks

    # 2. Metadata
    meta_start = time.perf_counter()
    meta = resolve_metadata(parsed_doc, filename)
    logger.info(
        "Metadata resolution took %.2fs",
        time.perf_counter() - meta_start
    )
    # 3. Enrichment — failures are isolated per block
    paper_title = meta.get("title", "")

    figure_blocks = sum(1 for b in blocks if b.content_type == "figure")
    table_blocks  = sum(1 for b in blocks if b.content_type == "table")
    logger.info(
        "Detected %d figures and %d tables",
        figure_blocks,
        table_blocks,
    )

    for block in blocks:
        if block.content_type == "figure" and block.image_path:
            try:
                caption_start = time.perf_counter()
                result = caption_figure(
                    block.image_path,
                    paper_title,
                    block.figure_number or "Figure",
                )
                logger.info(
                    "%s captioning took %.2fs",
                    block.figure_number,
                    time.perf_counter() - caption_start
                )

                original_caption = block.content or ""
                groq_caption = result["caption"]

                fallback_1 = f"{block.figure_number or 'Figure'} — image not available"
                fallback_2 = f"{block.figure_number or 'Figure'} from {paper_title}"

                # If Groq provided a real description, combine it with the original caption
                if groq_caption not in (fallback_1, fallback_2):
                    block.content = f"{original_caption}\n\nDescription: {groq_caption}".strip()
                    block.is_enriched = True
                else:
                    block.is_enriched = False

                if result.get("table_data"):
                    block.table_markdown = result["table_data"]
                    block.content_type   = "table"
            except Exception as e:
                logger.warning("Caption failed for %s: %s", block.figure_number, e)
                block.is_enriched = False

        elif block.content_type == "table" and block.table_markdown:
            try:
                summarize_start = time.perf_counter()
                summary = summarize_table(block.table_markdown, paper_title)
                logger.info(
                    "%s summarization took %.2fs",
                    block.figure_number or f"Table page {block.page_number}",
                    time.perf_counter() - summarize_start
                )

                if summary:
                    original_content = block.content or ""
                    block.content     = f"{original_content}\n\nSummary: {summary}".strip()
                    block.is_enriched = True
                    # Stash summary for the table_summary DB column via metadata
                    block.metadata["table_summary"] = summary
                else:
                    block.is_enriched = False
            except Exception as e:
                logger.warning("Table summary failed for %s: %s", block.figure_number, e)
                block.is_enriched = False

        else:
            # Text, equations, captions — structurally enriched by Docling itself
            block.is_enriched = True

    # 4. Chunk
    chunk_start = time.perf_counter()

    chunks = chunk_parsed_blocks(blocks, meta)
    logger.info(
        "Chunking took %.2fs",
        time.perf_counter() - chunk_start
    )
    logger.info(
        "Produced %d chunks",
        len(chunks),
    )

    # Sanitise null bytes — PostgreSQL text columns reject \x00
    for chunk in chunks:
        chunk["text"] = chunk["text"].replace("\x00", "")

    if not chunks:
        logger.info("_sync_parse_and_enrich took %.2fs", time.perf_counter() - func_start)
        return {"error": "No chunks produced after filtering",
                "chunks": [], "meta": meta, "filename": filename}

    logger.info("_sync_parse_and_enrich took %.2fs", time.perf_counter() - func_start)
    return {"chunks": chunks, "meta": meta, "filename": filename}


def _embed_batch_sync(batch: list[dict]) -> list:
    from src.retrieval.embedder import embed_chunks
    return embed_chunks(batch)


# ── Async orchestrator ────────────────────────────────────────────────────────

async def ingest_single_paper(pdf_path: str, db: AsyncSession) -> dict:
    """
    Full ingestion pipeline for one PDF.
    Returns a result summary dict on success or a dict with "error" key on failure.
    """
    loop = asyncio.get_running_loop()
    paper_start = time.perf_counter()

    logger.info("=" * 80)
    logger.info("Starting ingestion for: %s", pdf_path)
    parse_start = time.perf_counter()

    # CPU-bound parse + enrich off the event loop
    data = await loop.run_in_executor(None, _sync_parse_and_enrich, pdf_path)
    logger.info(
        "Parse + metadata + enrichment + chunking took %.2fs",
        time.perf_counter() - parse_start
    )

    if "error" in data and not data["chunks"]:
        logger.info(
            "TOTAL ingestion time: %.2fs",
            time.perf_counter() - paper_start,
        )
        logger.info("=" * 80)
        return data

    chunks   = data["chunks"]
    meta     = data["meta"]
    filename = data["filename"]

    # Normalise year to int (DB column is Integer)
    year = meta.get("year")
    try:
        year = int(year)
    except (TypeError, ValueError):
        year = None

    inserted  = 0
    fig_count = 0
    tbl_count = 0

    async with db.begin():

        # ── 1. Upsert Paper record ─────────────────────────────────────────────
        paper_res = await db.execute(text("""
            INSERT INTO papers (
                filename, title, authors, abstract, year, topic,
                url, doi, arxiv_id, chunk_count, indexed, created_at, updated_at
            ) VALUES (
                :filename, :title, :authors, :abstract, :year, :topic,
                :url, :doi, :arxiv_id, 0, false, now(), now()
            )
            ON CONFLICT (filename) DO UPDATE SET
                title      = EXCLUDED.title,
                authors    = EXCLUDED.authors,
                abstract   = EXCLUDED.abstract,
                year       = EXCLUDED.year,
                doi        = EXCLUDED.doi,
                arxiv_id   = EXCLUDED.arxiv_id,
                updated_at = now()
            RETURNING id
        """), {
            "filename": filename,
            "title":    meta.get("title", ""),
            "authors":  meta.get("authors", ""),
            "abstract": meta.get("abstract") or None,
            "year":     year,
            "topic":    meta.get("topic", ""),
            "url":      meta.get("url") or meta.get("arxiv_url"),
            "doi":      meta.get("doi"),
            "arxiv_id": meta.get("arxiv_id"),
        })
        paper_id = paper_res.scalar_one()

        # ── 2. Clear stale chunks for this paper ───────────────────────────────
        await db.execute(
            text("DELETE FROM chunks WHERE paper_id = :pid"),
            {"pid": paper_id},
        )
        
        # ── 3. Embed + insert in fixed-size batches ────────────────────────────
        embed_db_start = time.perf_counter()
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]

            embeddings = await loop.run_in_executor(None, _embed_batch_sync, batch)

            for chunk, emb in zip(batch, embeddings):
                m = chunk["metadata"]
                ct = m.get("content_type", "text")
                if ct == "figure":
                    fig_count += 1
                elif ct == "table":
                    tbl_count += 1

                vec = "[" + ",".join(str(x) for x in emb.tolist()) + "]"

                await db.execute(text("""
                    INSERT INTO chunks (
                        chunk_id, paper_id, source, title, authors, year,
                        section, section_priority, chunk_index,
                        chunk_type, content_type, text,
                        char_count, word_count,
                        page_number, figure_number, image_path,
                        table_markdown, table_summary, embedding, is_enriched
                    ) VALUES (
                        :chunk_id, :paper_id, :source, :title, :authors, :year,
                        :section, :section_priority, :chunk_index,
                        :chunk_type, :content_type, :text,
                        :char_count, :word_count,
                        :page_number, :figure_number, :image_path,
                        :table_markdown, :table_summary,
                        CAST(:embedding AS vector), :is_enriched
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text           = EXCLUDED.text,
                        table_summary  = EXCLUDED.table_summary,
                        embedding      = EXCLUDED.embedding,
                        section        = EXCLUDED.section,
                        is_enriched    = EXCLUDED.is_enriched
                """), {
                    "chunk_id":         m.get("chunk_id", ""),
                    "paper_id":         paper_id,
                    "source":           filename,
                    "title":            meta.get("title", ""),
                    "authors":          meta.get("authors", ""),
                    "year":             year,
                    "section":          m.get("section", ""),
                    "section_priority": m.get("section_priority", 0.5),
                    "chunk_index":      m.get("chunk_index", 0),
                    "chunk_type":       m.get("chunk_type", "general"),
                    "content_type":     ct,
                    "text":             chunk["text"],
                    "char_count":       m.get("char_count", 0),
                    "word_count":       m.get("word_count", 0),
                    "page_number":      m.get("page_number"),
                    "figure_number":    m.get("figure_number"),
                    "image_path":       m.get("image_path"),
                    "table_markdown":   m.get("table_markdown"),
                    "table_summary":    m.get("table_summary"),
                    "embedding":        vec,
                    "is_enriched":      m.get("is_enriched", False),
                })
                inserted += 1

            # Explicitly free embedding tensors before next batch
            del embeddings, batch

        logger.info(
            "Embedding + DB insertion took %.2fs",
            time.perf_counter() - embed_db_start
        )

        # ── 4. Finalise Paper stats ────────────────────────────────────────────
        await db.execute(text("""
            UPDATE papers SET
                chunk_count  = :c,
                figure_count = :f,
                table_count  = :t,
                indexed      = true,
                updated_at   = now()
            WHERE id = :pid
        """), {"c": inserted, "f": fig_count, "t": tbl_count, "pid": paper_id})

    # db.begin() commits here; rolls back automatically on any exception above.


    logger.info(
        "TOTAL ingestion time: %.2fs",
        time.perf_counter() - paper_start,
    )

    logger.info("=" * 80)
    return {
        "filename": filename,
        "title":    meta.get("title", ""),
        "chunks":   inserted,
        "figures":  fig_count,
        "tables":   tbl_count,
    }