"""
scripts/build_index.py
Full multimodal ingestion pipeline:
1. Parse PDFs (Docling → PyMuPDF fallback)
2. Extract figures, tables, text
3. Caption figures, summarize tables
4. Chunk everything
5. Embed with SPECTER2
6. Sync to pgvector
"""
import asyncio
import json
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

PAPERS_DIR    = "data/papers"
FIGURES_DIR   = "data/figures"
INDEXES_DIR   = "indexes"
CHUNKS_PATH   = "indexes/chunks_metadata.json"
EMBEDDINGS_PATH = "indexes/chunk_embeddings.npy"


def run_pipeline(force: bool = False, no_vlm: bool = False):
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(INDEXES_DIR).mkdir(parents=True, exist_ok=True)

    from src.ingestion.parser import parse_pdf
    from src.ingestion.chunker import chunk_parsed_blocks
    from src.ingestion.ingest import load_registry, extract_metadata_from_registry, extract_metadata_from_pdf
    from src.ingestion.table_summarizer import summarize_table
    from src.ingestion.figure_captioner import caption_figure, is_ollama_available
    from src.retrieval.embedder import embed_chunks, save_embeddings

    pdf_files = sorted(Path(PAPERS_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PAPERS_DIR}")
        return

    print(f"\nFound {len(pdf_files)} PDFs")
    ollama_ok = not no_vlm
    print(f"Figure captioning: {'enabled (DePlot + Groq)' if ollama_ok else 'disabled'}")

    all_chunks = []
    stats = {"papers": 0, "text": 0, "figures": 0, "tables": 0, "errors": 0}

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name[:60]}")
        try:
            # Parse
            blocks = parse_pdf(str(pdf_path), FIGURES_DIR)

            # Metadata
            filename = pdf_path.name
            meta = extract_metadata_from_registry(filename)
            if not meta:
                with open(pdf_path, "rb") as f:
                    import fitz
                    doc = fitz.open(str(pdf_path))
                    first_page = doc[0].get_text()
                    doc.close()
                from src.ingestion.ingest import extract_metadata_from_pdf
                meta = extract_metadata_from_pdf(first_page, filename)

            # Enhance figures and tables
            for block in blocks:
                if block.content_type == "figure" and block.image_path:
                    try:
                        result = caption_figure(
                            block.image_path,
                            meta.get("title", ""),
                            block.figure_number or "Figure"
                        )
                        block.content = result["caption"]
                        if result["table_data"]:
                            block.table_markdown = result["table_data"]
                            block.content_type = "table"
                            print(f"  Chart parsed → table: {block.figure_number}")
                    except Exception as e:
                        print(f"  Caption failed: {e}")

                elif block.content_type == "table" and block.table_markdown:
                    try:
                        summary = summarize_table(block.table_markdown, meta.get("title", ""))
                        if summary:
                            block.content = f"{block.content}\n\nSummary: {summary}"
                    except Exception as e:
                        print(f"  Table summary failed: {e}")

            # Chunk
            chunks = chunk_parsed_blocks(blocks, meta)

            # Count by type
            for c in chunks:
                ct = c["metadata"]["content_type"]
                stats[ct] = stats.get(ct, 0) + 1

            all_chunks.extend(chunks)
            stats["papers"] += 1

            fig_count = sum(1 for c in chunks if c["metadata"]["content_type"] == "figure")
            tbl_count = sum(1 for c in chunks if c["metadata"]["content_type"] == "table")
            txt_count = sum(1 for c in chunks if c["metadata"]["content_type"] == "text")
            print(f"  ✓ {len(chunks)} chunks — text:{txt_count} fig:{fig_count} tbl:{tbl_count}")

        except Exception as e:
            print(f"  FAILED: {e}")
            stats["errors"] += 1

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Stats: {stats}")

    # Save chunks
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Chunks saved → {CHUNKS_PATH}")

    # Embed
    print("\nEmbedding chunks...")
    embeddings = embed_chunks(all_chunks, batch_size=32)
    save_embeddings(embeddings, EMBEDDINGS_PATH)
    print(f"Embeddings saved → {EMBEDDINGS_PATH} {embeddings.shape}")

    return all_chunks, embeddings


async def sync_to_pgvector(all_chunks, embeddings):
    import asyncpg
    from datetime import datetime

    db_url = os.getenv("DATABASE_URL", "postgresql://anthology:anthology@localhost:5432/anthology")
    db_url_pg = db_url.replace("postgresql+asyncpg://", "postgresql://").replace("+asyncpg", "")

    print(f"\nSyncing {len(all_chunks)} chunks to pgvector...")
    conn = await asyncpg.connect(db_url_pg)

    # 1. Clear existing data to start fresh for bulk build
    await conn.execute("DELETE FROM chunks")
    await conn.execute("DELETE FROM papers")

    now = datetime.utcnow()

    # Group chunks by source for relational insertion
    from collections import defaultdict
    paper_groups = defaultdict(list)
    for i, chunk in enumerate(all_chunks):
        paper_groups[chunk["metadata"]["source"]].append(i)

    for filename, indices in paper_groups.items():
        # Representative chunk for metadata
        rep_chunk = all_chunks[indices[0]]
        meta = rep_chunk["metadata"]

        # a. Create Paper record
        paper_id = await conn.fetchval("""
            INSERT INTO papers (filename, title, authors, year, topic, url, chunk_count, indexed, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, 0, false, $7, $7)
            RETURNING id
        """,
        filename, meta["title"], meta.get("authors", ""),
        int(meta["year"]) if meta.get("year") else None,
        meta.get("topic", ""), meta.get("url", ""), now)

        # b. Insert associated chunks
        fig_count = 0
        tbl_count = 0
        for idx in indices:
            chunk = all_chunks[idx]
            emb = embeddings[idx]
            m = chunk["metadata"]
            vec = "[" + ",".join(str(x) for x in emb.tolist()) + "]"

            if m.get("content_type") == "figure": fig_count += 1
            elif m.get("content_type") == "table": tbl_count += 1

            await conn.execute(f"""
                INSERT INTO chunks (
                    id, chunk_id, paper_id, source, title, authors, year,
                    section, section_priority, chunk_index, chunk_type, content_type,
                    text, char_count, word_count,
                    page_number, figure_number, image_path, table_markdown, table_summary,
                    embedding, created_at
                ) VALUES (
                    gen_random_uuid(), $1, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10, $11,
                    $12, $13, $14,
                    $15, $16, $17, $18, $19,
                    '{vec}'::vector, $20
                )
            """,
            m["chunk_id"], paper_id, m["source"], m["title"],
            m.get("authors", ""),
            int(m["year"]) if m.get("year") else None,
            m.get("section", ""),
            float(m.get("section_priority", 0.5)),
            int(m.get("chunk_index", 0)),
            m.get("chunk_type", "general"),
            m.get("content_type", "text"),
            chunk["text"].replace("\x00", ""),
            int(m.get("char_count", 0)),
            int(m.get("word_count", 0)),
            m.get("page_number"),
            m.get("figure_number"),
            m.get("image_path"),
            m.get("table_markdown"),
            m.get("table_summary"),
            now,
            )

        # c. Update Paper stats
        await conn.execute("""
            UPDATE papers SET
                chunk_count = $1, figure_count = $2, table_count = $3, indexed = true, updated_at = $4
            WHERE id = $5
        """, len(indices), fig_count, tbl_count, now, paper_id)

    await conn.close()
    print(f"Done: {len(all_chunks)} chunks and {len(paper_groups)} papers in pgvector")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(force=args.force, no_vlm=args.no_vlm)
    if result and not args.no_sync:
        all_chunks, embeddings = result
        asyncio.run(sync_to_pgvector(all_chunks, embeddings))
