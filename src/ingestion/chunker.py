from __future__ import annotations

import hashlib
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Section configuration ─────────────────────────────────────────────────────

SKIP_SECTIONS: set[str] = {
    "references", "appendix", "acknowledgements", "acknowledgments",
    "bibliography", "funding", "conflict of interest", "author contributions",
    "declaration of competing interest", "data availability",
}

SECTION_PRIORITY: dict[str, float] = {
    "abstract":           1.0,
    "introduction":       0.9,
    "methodology":        1.0,
    "method":             1.0,
    "methods":            1.0,
    "approach":           1.0,
    "proposed method":    1.0,
    "architecture":       1.0,
    "model":              0.9,
    "results":            0.95,
    "evaluation":         0.95,
    "experiment":         0.85,
    "experiments":        0.85,
    "experimental setup": 0.85,
    "discussion":         0.8,
    "conclusion":         0.75,
    "conclusions":        0.75,
    "background":         0.7,
    "related work":       0.65,
    "literature review":  0.65,
    "preamble":           0.5,
}

# Sections where equations and math-dense text should use a larger chunk window
_MATH_SECTIONS: set[str] = {
    "methodology", "method", "methods", "approach",
    "model", "architecture", "proposed method",
}

# ── Chunk size constants ───────────────────────────────────────────────────────

_CHUNK_SIZE_DEFAULT = 1400
_CHUNK_SIZE_MATH    = 1800
_CHUNK_OVERLAP      = 400
_TABLE_CHUNK_SIZE   = 2000
_TABLE_CHUNK_OVERLAP = 200

# Minimum content thresholds — below these a chunk is skipped unless it contains
# a valuable short fact (metric, stat, p-value).
_MIN_TEXT_CHUNK   = 60    # chars
_MIN_TABLE_CHUNK  = 20    # chars — tables can be dense with numbers
_MIN_BLOCK_CONTENT = 50   # chars — pre-split threshold


# ── Utilities ─────────────────────────────────────────────────────────────────

def _chunk_id(source: str, content_type: str, block_idx: int, sub_idx: int) -> str:
    raw = f"{source}::{content_type}::{block_idx}::{sub_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _section_priority(section: str) -> float:
    return SECTION_PRIORITY.get(section.lower(), 0.6)


def _should_skip_section(section: str) -> bool:
    return section.lower() in SKIP_SECTIONS


def is_valuable_short_fact(text: str) -> bool:
    """True when text contains scientific metrics worth keeping even if short."""
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


def _make_text_splitter(section: str) -> RecursiveCharacterTextSplitter:
    size = _CHUNK_SIZE_MATH if section.lower() in _MATH_SECTIONS else _CHUNK_SIZE_DEFAULT
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )


def _make_table_splitter() -> RecursiveCharacterTextSplitter:
    """Split on newlines to keep individual rows intact."""
    return RecursiveCharacterTextSplitter(
        chunk_size=_TABLE_CHUNK_SIZE,
        chunk_overlap=_TABLE_CHUNK_OVERLAP,
        separators=["\n", " ", ""],
        length_function=len,
    )


def _table_header_prefix(table_text: str) -> str:
    """
    Return the markdown header + separator rows from a markdown table.
    Prepended to every split chunk beyond the first so the column names
    are always present for retrieval context.
    """
    lines = table_text.splitlines()
    if len(lines) < 2:
        return ""
    sep_re = re.compile(r"^\|[\s\-\|:]+\|?\s*$")
    if sep_re.match(lines[1].strip()):
        return lines[0] + "\n" + lines[1] + "\n"
    return ""


# ── Main entry point ──────────────────────────────────────────────────────────

def chunk_parsed_blocks(blocks: list, metadata: dict) -> list[dict]:
    """
    Convert a list of ParsedBlock objects into chunk dicts ready for embedding
    and DB insertion.

    Parameters
    ----------
    blocks   : list[ParsedBlock] — from parser.parse_pdf_document().blocks
    metadata : dict              — resolved paper metadata (title, authors, year …)

    Returns
    -------
    list[dict] with keys "text" and "metadata"
    """
    source = metadata.get("filename", "unknown")
    chunks: list[dict] = []
    global_chunk_idx = 0


    for block_idx, block in enumerate(blocks):
        ct = block.content_type  # text | table | figure | equation | caption

        # ── Skip low-value sections ────────────────────────────────────────────
        if _should_skip_section(block.section_title or ""):
            continue

        sec      = block.section_title or ""
        priority = _section_priority(sec)
        prefix   = f"[{sec}] " if sec else ""

        # ── Figure ────────────────────────────────────────────────────────────
        if ct == "figure":
            # Skip figures without a saved image — no content to embed meaningfully
            if not block.image_path:
                continue

            chunk_text = block.content or f"{block.figure_number or 'Figure'} from {metadata.get('title', '')}"
            full_text  = prefix + chunk_text

            chunks.append(_make_chunk(
                text       = full_text,
                source     = source,
                metadata   = metadata,
                block      = block,
                chunk_id   = _chunk_id(source, "figure", block_idx, 0),
                sec        = sec,
                priority   = priority,
                chunk_idx  = global_chunk_idx,
                chunk_type = "figure_table",
                ct         = "figure",
                raw_text   = chunk_text,
            ))
            global_chunk_idx += 1

        # ── Table ─────────────────────────────────────────────────────────────
        elif ct == "table":
            # Use the embeddable content built by parser._table_to_embeddable,
            # which already has column names + cell values in readable form.
            # Fall back to raw markdown only if content is empty.
            chunk_text = block.content or block.table_markdown or ""
            if not chunk_text.strip():
                continue
            if len(chunk_text.strip()) < _MIN_TABLE_CHUNK and not is_valuable_short_fact(chunk_text):
                continue

            if len(chunk_text) <= _TABLE_CHUNK_SIZE:
                # Fits in one chunk — no splitting needed
                full_text = prefix + chunk_text
                chunks.append(_make_chunk(
                    text          = full_text,
                    source        = source,
                    metadata      = metadata,
                    block         = block,
                    chunk_id      = _chunk_id(source, "table", block_idx, 0),
                    sec           = sec,
                    priority      = priority,
                    chunk_idx     = global_chunk_idx,
                    chunk_type    = "quantitative",
                    ct            = "table",
                    raw_text      = chunk_text,
                    table_md      = block.table_markdown,
                    table_summary = block.metadata.get("table_summary"),
                ))
                global_chunk_idx += 1
            else:
                # Large table — split on row boundaries, re-attach header to each chunk
                splitter      = _make_table_splitter()
                splits        = splitter.split_text(chunk_text)
                header_prefix = _table_header_prefix(chunk_text)

                for ti, t_split in enumerate(splits):
                    if len(t_split.strip()) < _MIN_TABLE_CHUNK and not is_valuable_short_fact(t_split):
                        continue

                    # First split already has the header; subsequent splits get it prepended
                    text_with_header = (
                        t_split if ti == 0 or not header_prefix
                        else header_prefix + t_split
                    )
                    full_text = prefix + text_with_header

                    chunks.append(_make_chunk(
                        text          = full_text,
                        source        = source,
                        metadata      = metadata,
                        block         = block,
                        chunk_id      = _chunk_id(source, "table", block_idx, ti),
                        sec           = sec,
                        priority      = priority,
                        chunk_idx     = global_chunk_idx,
                        chunk_type    = "quantitative",
                        ct            = "table",
                        raw_text      = t_split,
                        # Only first split carries the full markdown (avoids redundant storage)
                        table_md      = block.table_markdown if ti == 0 else None,
                        table_summary = block.metadata.get("table_summary") if ti == 0 else None,
                    ))
                    global_chunk_idx += 1

        # ── Equation ──────────────────────────────────────────────────────────
        elif ct == "equation":
            # Equations are NEVER split — mathematical notation loses meaning
            # if broken across chunks. Keep whole regardless of length.
            content = block.content.strip()
            if not content:
                continue
            full_text = prefix + content
            chunks.append(_make_chunk(
                text       = full_text,
                source     = source,
                metadata   = metadata,
                block      = block,
                chunk_id   = _chunk_id(source, "equation", block_idx, 0),
                sec        = sec,
                priority   = priority,
                chunk_idx  = global_chunk_idx,
                chunk_type = "math_heavy",
                ct         = "equation",
                raw_text   = content,
            ))
            global_chunk_idx += 1

        # ── Text / Caption ────────────────────────────────────────────────────
        elif ct in ("text", "caption"):
            content = block.content.strip()
            if len(content) < _MIN_BLOCK_CONTENT and not is_valuable_short_fact(content):
                continue

            splitter = _make_text_splitter(sec)
            splits   = splitter.split_text(content)

            for si, split in enumerate(splits):
                split = split.strip()
                if len(split) < _MIN_TEXT_CHUNK and not is_valuable_short_fact(split):
                    continue

                full_text = prefix + split
                chunks.append(_make_chunk(
                    text       = full_text,
                    source     = source,
                    metadata   = metadata,
                    block      = block,
                    chunk_id   = _chunk_id(source, ct, block_idx, si),
                    sec        = sec,
                    priority   = priority,
                    chunk_idx  = global_chunk_idx,
                    chunk_type = detect_chunk_type(split),
                    ct         = "text",   # caption → stored as text in DB
                    raw_text   = split,
                ))
                global_chunk_idx += 1

    return chunks


# ── Chunk factory ─────────────────────────────────────────────────────────────

def _make_chunk(
    *,
    text:          str,
    source:        str,
    metadata:      dict,
    block,                  # ParsedBlock
    chunk_id:      str,
    sec:           str,
    priority:      float,
    chunk_idx:     int,
    chunk_type:    str,
    ct:            str,
    raw_text:      str,
    table_md:      str | None = None,
    table_summary: str | None = None,
) -> dict:
    """
    Build the canonical chunk dict consumed by ingest_service.py.
    All DB column keys are present — none are optional here so INSERT
    never hits a KeyError.
    """
    return {
        "text": text,
        "metadata": {
            "chunk_id":         chunk_id,
            "source":           source,
            "title":            metadata.get("title", ""),
            "authors":          metadata.get("authors", ""),
            "year":             metadata.get("year"),
            "section":          sec,
            "section_priority": priority,
            "chunk_index":      chunk_idx,
            "chunk_type":       chunk_type,
            "content_type":     ct,
            "char_count":       len(raw_text),
            "word_count":       len(raw_text.split()),
            "page_number":      block.page_number,
            "figure_number":    block.figure_number,
            "image_path":       block.image_path,
            "table_markdown":   table_md,
            "table_summary":    table_summary,  # FIX: previously hardcoded None —
                                                 # ingest_service.py stashes the
                                                 # Groq-generated summary at
                                                 # block.metadata["table_summary"];
                                                 # this now forwards it instead of
                                                 # silently dropping it before INSERT.
            "is_enriched":      block.is_enriched,
        },
    }