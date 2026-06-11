"""
Chunk text blocks. Tables and figures are stored as single chunks.
"""
from __future__ import annotations
from multimodal.ingestion.parser import ParsedBlock

MAX_CHARS = 1400
OVERLAP_CHARS = 200


def chunk_blocks(blocks: list[ParsedBlock], paper_metadata: dict) -> list[dict]:
    """
    Convert ParsedBlocks into chunk dicts ready for embedding + storage.
    Text blocks are split if > MAX_CHARS.
    Tables and figures are kept as single chunks.
    """
    chunks = []

    for block in blocks:
        if block.content_type == "text":
            text_chunks = _split_text(block.content)
            for i, text in enumerate(text_chunks):
                chunks.append({
                    "content_type": "text",
                    "content": text,
                    "page_number": block.page_number,
                    "section_title": block.section_title,
                    "figure_number": None,
                    "image_path": None,
                    "table_markdown": None,
                    "table_summary": None,
                    "metadata": {**paper_metadata, "chunk_index": i},
                })

        elif block.content_type == "table":
            chunks.append({
                "content_type": "table",
                "content": block.content,
                "page_number": block.page_number,
                "section_title": block.section_title,
                "figure_number": block.figure_number,
                "image_path": None,
                "table_markdown": block.table_markdown,
                "table_summary": None,  # filled by summarizer
                "metadata": paper_metadata,
            })

        elif block.content_type == "figure":
            chunks.append({
                "content_type": "figure",
                "content": block.content,  # caption or placeholder
                "page_number": block.page_number,
                "section_title": block.section_title,
                "figure_number": block.figure_number,
                "image_path": block.image_path,
                "table_markdown": None,
                "table_summary": None,
                "metadata": paper_metadata,
            })

    return chunks


def _split_text(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHARS
        if end < len(text):
            # break at last sentence boundary
            boundary = text.rfind(". ", start, end)
            if boundary > start:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - OVERLAP_CHARS
    return [c for c in chunks if len(c) > 50]
