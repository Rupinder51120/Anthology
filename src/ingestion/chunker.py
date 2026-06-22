import re
import json
import hashlib
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


SKIP_SECTIONS = {
    "references", "appendix", "acknowledgements", "acknowledgments",
    "bibliography", "funding", "conflict of interest", "author contributions",
}

SECTION_PRIORITY = {
    "abstract":        1.0,
    "introduction":    0.9,
    "methodology":     1.0,
    "method":          1.0,
    "methods":         1.0,
    "approach":        1.0,
    "proposed method": 1.0,
    "architecture":    1.0,
    "model":           0.9,
    "results":         0.95,
    "evaluation":      0.95,
    "experiment":      0.85,
    "experiments":     0.85,
    "discussion":      0.8,
    "conclusion":      0.75,
    "background":      0.7,
    "related work":    0.65,
    "literature review": 0.65,
    "preamble":        0.5,
}

MATH_SECTIONS = {"methodology", "method", "methods", "approach", "model", "architecture"}
CHUNK_SIZE_DEFAULT = 1400
CHUNK_SIZE_MATH    = 1800
CHUNK_OVERLAP      = 200

TABLE_CHUNK_SIZE = 2000
TABLE_CHUNK_OVERLAP = 100


def _chunk_id(source: str, section: str, index: int) -> str:
    raw = f"{source}::{section}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _make_table_splitter() -> RecursiveCharacterTextSplitter:
    # Split on markdown table row boundaries first, falling back to lines.
    # This keeps individual rows intact rather than slicing mid-row.
    return RecursiveCharacterTextSplitter(
        chunk_size=TABLE_CHUNK_SIZE,
        chunk_overlap=TABLE_CHUNK_OVERLAP,
        separators=["\n", " ", ""],
        length_function=len,
    )


def _table_header_prefix(table_text: str) -> str:
    """
    Return the markdown header block (column-name row + separator row) from
    a markdown table, or an empty string if the table has no recognisable
    header.
    """
    lines = table_text.splitlines()
    if len(lines) < 2:
        return ""

    sep_re = re.compile(r"^\|[\s\-\|:]+\|?\s*$")
    if sep_re.match(lines[1].strip()):
        return lines[0] + "\n" + lines[1] + "\n"
    return ""


def is_valuable_short_fact(text: str) -> bool:
    """
    Returns True if the text contains patterns typical of scientific facts,
    metrics, or statistical results, justifying its preservation even if short.
    """
    if re.search(r'\d+\.\d+|\d+%|p\s*[<>=]\s*0\.\d+', text):
        return True
    if re.search(r'\b[A-Z]{2,}\s*[=:]\s*[-+]?\d*\.?\d+', text):
        return True
    return False


def detect_chunk_type(text: str) -> str:
    math_patterns = [
        r'\$.*?\$', r'\\[a-zA-Z]+\{', r'[αβγδεζηθλμπσφψω]',
        r'=\s*[\d\\]', r'\\frac', r'\\sum', r'\\int', r'\^{', r'_{',
    ]
    if any(re.search(p, text) for p in math_patterns):
        return "math_heavy"
    if re.search(r'\b(table|figure|fig\.)\s+\d+', text, re.IGNORECASE):
        return "figure_table"
    if re.search(r'\d+\.\d+|\d+%|p\s*[<>=]\s*0\.\d+', text):
        return "quantitative"
    if len(text.split()) > 80:
        return "narrative"
    return "general"


def _make_splitter(section_name: str) -> RecursiveCharacterTextSplitter:
    size = CHUNK_SIZE_MATH if section_name.lower() in MATH_SECTIONS else CHUNK_SIZE_DEFAULT
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )


def chunk_paper(paper: dict) -> list[dict]:
    """Chunk a paper dict from ingest.py (text-only path)."""
    metadata = paper["metadata"]
    source   = metadata["filename"]
    chunks   = []

    if paper.get("sections"):
        for section_name, section_text in paper["sections"].items():
            if section_name.lower() in SKIP_SECTIONS:
                continue
            if not section_text or (len(section_text.strip()) < 50 and not is_valuable_short_fact(section_text)):
                continue

            splitter = _make_splitter(section_name)
            splits   = splitter.split_text(section_text)
            priority = SECTION_PRIORITY.get(section_name.lower(), 0.6)

            for i, split in enumerate(splits):
                if len(split.strip()) < 60 and not is_valuable_short_fact(split):
                    continue
                chunk_type = detect_chunk_type(split)
                chunks.append({
                    "text": split,
                    "metadata": {
                        "chunk_id":         _chunk_id(source, section_name, i),
                        "source":           source,
                        "title":            metadata["title"],
                        "authors":          metadata["authors"],
                        "year":             metadata["year"],
                        "arxiv_id":         metadata.get("arxiv_id"),
                        "doi":              metadata.get("doi"),
                        "section":          section_name,
                        "section_priority": priority,
                        "chunk_index":      i,
                        "chunk_type":       chunk_type,
                        "content_type":     "text",
                        "char_count":       len(split),
                        "word_count":       len(split.split()),
                        "page_number":      None,
                        "figure_number":    None,
                        "image_path":       None,
                        "table_markdown":   None,
                        "table_summary":    None,
                        "is_enriched":      True,
                    }
                })
    else:
        for page in paper.get("pages", []):
            if not page["text"].strip():
                continue
            splitter = _make_splitter("default")
            splits   = splitter.split_text(page["text"])
            for i, split in enumerate(splits):
                if len(split.strip()) < 60 and not is_valuable_short_fact(split):
                    continue
                section_label = f"page_{page['page']}"
                chunks.append({
                    "text": split,
                    "metadata": {
                        "chunk_id":         _chunk_id(source, section_label, i),
                        "source":           source,
                        "title":            metadata["title"],
                        "authors":          metadata["authors"],
                        "year":             metadata["year"],
                        "arxiv_id":         metadata.get("arxiv_id"),
                        "doi":              metadata.get("doi"),
                        "section":          section_label,
                        "section_priority": 0.5,
                        "chunk_index":      i,
                        "chunk_type":       detect_chunk_type(split),
                        "content_type":     "text",
                        "char_count":       len(split),
                        "word_count":       len(split.split()),
                        "page_number":      page["page"],
                        "figure_number":    None,
                        "image_path":       None,
                        "table_markdown":   None,
                        "table_summary":    None,
                        "is_enriched":      True,
                    }
                })

    return chunks


def chunk_parsed_blocks(blocks: list, metadata: dict) -> list[dict]:

    source  = metadata["filename"]
    chunks  = []
    txt_idx = 0
    block_idx = 0  # guarantees globally-unique chunk_ids within this paper

    for block in blocks:
        ct = block.content_type  # text | figure | table | equation | caption

        if ct == "figure":
            if not block.image_path:
                continue
            chunk_text = block.content or f"{block.figure_number or 'Figure'} from {metadata['title']}"
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_id":         _chunk_id(source, "figure", block_idx),
                    "source":           source,
                    "title":            metadata["title"],
                    "authors":          metadata.get("authors", ""),
                    "year":             metadata.get("year"),
                    "section":          block.section_title or "",
                    "section_priority": SECTION_PRIORITY.get((block.section_title or "").lower(), 0.6),
                    "chunk_index":      0,
                    "chunk_type":       "figure_table",
                    "content_type":     "figure",
                    "char_count":       len(chunk_text),
                    "word_count":       len(chunk_text.split()),
                    "page_number":      block.page_number,
                    "figure_number":    block.figure_number,
                    "image_path":       block.image_path,
                    "table_markdown":   None,
                    "table_summary":    None,
                    "is_enriched":      block.is_enriched,
                }
            })
            block_idx += 1

        elif ct == "table":
            chunk_text = block.content or block.table_markdown or ""
            if len(chunk_text.strip()) < 20 and not is_valuable_short_fact(chunk_text):
                continue

            if len(chunk_text) <= TABLE_CHUNK_SIZE:
                # Small enough — keep as a single chunk, same as before.
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id":         _chunk_id(source, "table", block_idx),
                        "source":           source,
                        "title":            metadata["title"],
                        "authors":          metadata.get("authors", ""),
                        "year":             metadata.get("year"),
                        "section":          block.section_title or "",
                        "section_priority": SECTION_PRIORITY.get((block.section_title or "").lower(), 0.6),
                        "chunk_index":      0,
                        "chunk_type":       "quantitative",
                        "content_type":     "table",
                        "char_count":       len(chunk_text),
                        "word_count":       len(chunk_text.split()),
                        "page_number":      block.page_number,
                        "figure_number":    block.figure_number,
                        "image_path":       None,
                        "table_markdown":   block.table_markdown,
                        "table_summary":    None,
                        "is_enriched":      block.is_enriched,
                    }
                })
                block_idx += 1
            else:

                table_splitter = _make_table_splitter()
                table_splits = table_splitter.split_text(chunk_text)

                header_prefix = _table_header_prefix(chunk_text)
                for ti, t_split in enumerate(table_splits):
                    if len(t_split.strip()) < 20 and not is_valuable_short_fact(t_split):
                        continue

                    text_with_header = (
                        t_split if ti == 0 or not header_prefix
                        else header_prefix + t_split
                    )
                    chunks.append({
                        "text": text_with_header,
                        "metadata": {
                            "chunk_id":         _chunk_id(source, "table", block_idx),
                            "source":           source,
                            "title":            metadata["title"],
                            "authors":          metadata.get("authors", ""),
                            "year":             metadata.get("year"),
                            "section":          block.section_title or "",
                            "section_priority": SECTION_PRIORITY.get((block.section_title or "").lower(), 0.6),
                            "chunk_index":      ti,
                            "chunk_type":       "quantitative",
                            "content_type":     "table",
                            "char_count":       len(t_split),
                            "word_count":       len(t_split.split()),
                            "page_number":      block.page_number,
                            "figure_number":    block.figure_number,
                            "image_path":       None,
                            "table_markdown":   block.table_markdown if ti == 0 else None,
                            "table_summary":    None,
                            "is_enriched":      block.is_enriched,
                        }
                    })
                    block_idx += 1
        elif ct in ("text", "equation", "caption"):
            if len(block.content.strip()) < 50 and not is_valuable_short_fact(block.content):
                continue
            splitter = _make_splitter(block.section_title or "")
            splits   = splitter.split_text(block.content)
            priority = SECTION_PRIORITY.get((block.section_title or "").lower(), 0.6)

            for i, split in enumerate(splits):
                if len(split.strip()) < 60 and not is_valuable_short_fact(split):
                    continue
                chunks.append({
                    "text": split,
                    "metadata": {
                        "chunk_id":         _chunk_id(source, "text", block_idx),
                        "source":           source,
                        "title":            metadata["title"],
                        "authors":          metadata.get("authors", ""),
                        "year":             metadata.get("year"),
                        "section":          block.section_title or "",
                        "section_priority": priority,
                        "chunk_index":      txt_idx,
                        "chunk_type":       detect_chunk_type(split),
                        "content_type":     ct if ct == "equation" else "text",
                        "char_count":       len(split),
                        "word_count":       len(split.split()),
                        "page_number":      block.page_number,
                        "figure_number":    None,
                        "image_path":       None,
                        "table_markdown":   None,
                        "table_summary":    None,
                        "is_enriched":      block.is_enriched,
                    }
                })
                txt_idx += 1
                block_idx += 1

    return chunks
