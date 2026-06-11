"""
Main ingestion pipeline.
PDF → Parse → Chunk → Summarize → Caption → Embed → Store
"""
from __future__ import annotations
import uuid
import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from multimodal.ingestion.parser import parse_pdf
from multimodal.ingestion.chunker import chunk_blocks
from multimodal.ingestion.table_summarizer import summarize_table
from multimodal.ingestion.figure_captioner import caption_figure, is_ollama_available
from multimodal.ingestion.embedder import embed_texts


async def ingest_paper(
    pdf_path: str,
    paper_metadata: dict,
    session: AsyncSession,
    use_vlm: bool = True,
) -> dict:
    """
    Full ingestion pipeline for one PDF.
    Returns stats dict.
    """
    stats = {"text": 0, "table": 0, "figure": 0, "errors": 0}
    ollama_ok = is_ollama_available() if use_vlm else False

    print(f"Parsing {pdf_path}...")
    blocks = parse_pdf(pdf_path)
    print(f"  → {len(blocks)} blocks extracted")

    chunks = chunk_blocks(blocks, paper_metadata)
    print(f"  → {len(chunks)} chunks after splitting")

    # Insert paper record
    paper_id = str(uuid.uuid4())
    result = await session.execute(text("""
        INSERT INTO papers (id, filename, title, authors, abstract, year, arxiv_id, indexed, chunk_count, figure_count, table_count, created_at)
        VALUES (:id, :filename, :title, :authors, :abstract, :year, :arxiv_id, false, 0, 0, 0, NOW())
        ON CONFLICT (filename) DO UPDATE SET title = EXCLUDED.title
        RETURNING id
    """), {
        "id": paper_id,
        "filename": paper_metadata.get("filename", Path(pdf_path).name),
        "title": paper_metadata.get("title", "Unknown"),
        "authors": paper_metadata.get("authors", ""),
        "abstract": paper_metadata.get("abstract", ""),
        "year": paper_metadata.get("year"),
        "arxiv_id": paper_metadata.get("arxiv_id"),
    })
    paper_id = str(result.scalar())
    await session.commit()

    # Process each chunk
    texts_to_embed = []
    processed = []

    for chunk in chunks:
        try:
            # Table: generate summary via Groq
            if chunk["content_type"] == "table" and chunk["table_markdown"]:
                summary = summarize_table(
                    chunk["table_markdown"],
                    paper_metadata.get("title", "")
                )
                chunk["table_summary"] = summary
                chunk["content"] = f"{chunk['content']}\n\nSummary: {summary}" if summary else chunk["content"]

            # Figure: generate caption via VLM
            elif chunk["content_type"] == "figure" and ollama_ok:
                caption = caption_figure(
                    chunk["image_path"],
                    paper_metadata.get("title", ""),
                    chunk["figure_number"] or "Figure",
                )
                chunk["content"] = caption

            texts_to_embed.append(chunk["content"])
            processed.append(chunk)

        except Exception as e:
            print(f"  Chunk processing error: {e}")
            stats["errors"] += 1

    # Batch embed
    print(f"  Embedding {len(texts_to_embed)} chunks...")
    embeddings = embed_texts(texts_to_embed)

    # Store all chunks
    for chunk, emb in zip(processed, embeddings):
        await session.execute(text("""
            INSERT INTO chunks (
                id, paper_id, content_type, content,
                page_number, section_title, figure_number,
                image_path, table_markdown, table_summary,
                embedding, extra_metadata, created_at
            ) VALUES (
                :id, :paper_id, :content_type, :content,
                :page_number, :section_title, :figure_number,
                :image_path, :table_markdown, :table_summary,
                :embedding, :extra_metadata, NOW()
            )
        """), {
            "id": str(uuid.uuid4()),
            "paper_id": paper_id,
            "content_type": chunk["content_type"],
            "content": chunk["content"],
            "page_number": chunk["page_number"],
            "section_title": chunk["section_title"],
            "figure_number": chunk["figure_number"],
            "image_path": chunk["image_path"],
            "table_markdown": chunk.get("table_markdown"),
            "table_summary": chunk.get("table_summary"),
            "embedding": "[" + ",".join(str(x) for x in emb.tolist()) + "]",
            "extra_metadata": __import__("json").dumps(chunk.get("metadata", {})),
        })
        stats[chunk["content_type"]] = stats.get(chunk["content_type"], 0) + 1

    await session.commit()

    # Update paper stats
    await session.execute(text("""
        UPDATE papers SET
            chunk_count = :total,
            figure_count = :figures,
            table_count = :tables,
            indexed = true
        WHERE id = :paper_id
    """), {
        "total": len(processed),
        "figures": stats.get("figure", 0),
        "tables": stats.get("table", 0),
        "paper_id": paper_id,
    })
    await session.commit()

    print(f"  ✓ Ingested: {stats}")
    return stats
