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


def _chunk_id(source: str, section: str, index: int) -> str:
    raw = f"{source}::{section}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


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
            if not section_text or len(section_text.strip()) < 50:
                continue

            splitter = _make_splitter(section_name)
            splits   = splitter.split_text(section_text)
            priority = SECTION_PRIORITY.get(section_name.lower(), 0.6)

            for i, split in enumerate(splits):
                if len(split.strip()) < 60:
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
                    }
                })
    else:
        for page in paper.get("pages", []):
            if not page["text"].strip():
                continue
            splitter = _make_splitter("default")
            splits   = splitter.split_text(page["text"])
            for i, split in enumerate(splits):
                if len(split.strip()) < 60:
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
                    }
                })

    return chunks


def chunk_parsed_blocks(blocks: list, metadata: dict) -> list[dict]:
    """
    Chunk ParsedBlock objects from parser.py (multimodal path).
    Handles text, figure, table, equation blocks.
    """
    source  = metadata["filename"]
    chunks  = []
    txt_idx = 0

    for block in blocks:
        ct = block.content_type  # text | figure | table | equation | caption

        if ct == "figure":
            if not block.image_path:
                continue
            chunk_text = block.content or f"{block.figure_number or 'Figure'} from {metadata['title']}"
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_id":         _chunk_id(source, f"figure_{block.figure_number}", 0),
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
                }
            })

        elif ct == "table":
            chunk_text = block.content or block.table_markdown or ""
            if len(chunk_text.strip()) < 20:
                continue
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_id":         _chunk_id(source, f"table_{block.figure_number}", 0),
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
                }
            })

        elif ct in ("text", "equation", "caption"):
            if len(block.content.strip()) < 50:
                continue
            splitter = _make_splitter(block.section_title or "")
            splits   = splitter.split_text(block.content)
            priority = SECTION_PRIORITY.get((block.section_title or "").lower(), 0.6)

            for i, split in enumerate(splits):
                if len(split.strip()) < 60:
                    continue
                chunks.append({
                    "text": split,
                    "metadata": {
                        "chunk_id":         _chunk_id(source, block.section_title or "text", txt_idx),
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
                    }
                })
                txt_idx += 1

    return chunks


def chunk_all_papers(all_papers: list[dict]) -> list[dict]:
    all_chunks = []
    for paper in all_papers:
        paper_chunks = chunk_paper(paper)
        all_chunks.extend(paper_chunks)
        types = {}
        for c in paper_chunks:
            t = c["metadata"]["chunk_type"]
            types[t] = types.get(t, 0) + 1
        print(f"  {paper['metadata']['filename'][:45]}: {len(paper_chunks)} chunks | {types}")
    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: list[dict], output_path: str = "indexes/chunks_metadata.json"):
    Path("indexes").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Chunks saved → {output_path}")
