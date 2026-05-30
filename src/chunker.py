import re
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


SKIP_SECTIONS = {"references", "appendix"}

# section priority for retrieval (higher = more important)
SECTION_PRIORITY = {
    "abstract": 1.0,
    "introduction": 0.9,
    "methodology": 1.0,
    "method": 1.0,
    "methods": 1.0,
    "approach": 1.0,
    "proposed method": 1.0,
    "architecture": 1.0,
    "model": 0.9,
    "results": 0.95,
    "evaluation": 0.95,
    "experiment": 0.85,
    "experiments": 0.85,
    "discussion": 0.8,
    "conclusion": 0.75,
    "background": 0.7,
    "related work": 0.65,
    "literature review": 0.65,
    "preamble": 0.5,
}


def detect_chunk_type(text: str) -> str:
    """Classify chunk content type for metadata."""
    text_lower = text.lower()

    math_patterns = [r'\$.*?\$', r'\\[a-zA-Z]+\{', r'[αβγδεζηθλμπσφψω]',
                     r'=\s*[\d\\]', r'\\frac', r'\\sum', r'\\int', r'\^{', r'_{']
    if any(re.search(p, text) for p in math_patterns):
        return "math_heavy"

    if re.search(r'\b(table|figure|fig\.)\s+\d+', text_lower):
        return "figure_table"

    if re.search(r'\d+\.\d+|\d+%|p\s*[<>=]\s*0\.\d+', text):
        return "quantitative"

    if len(text.split()) > 80:
        return "narrative"

    return "general"


def chunk_paper(paper: dict,
                chunk_size: int = 512,
                chunk_overlap: int = 100) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len
    )

    metadata = paper["metadata"]
    chunks   = []

    if paper["sections"]:
        for section_name, section_text in paper["sections"].items():
            if section_name in SKIP_SECTIONS:
                continue
            if not section_text or len(section_text.strip()) < 50:
                continue

            splits = splitter.split_text(section_text)
            priority = SECTION_PRIORITY.get(section_name, 0.6)

            for i, split in enumerate(splits):
                if len(split.strip()) < 40:
                    continue

                chunk_type = detect_chunk_type(split)

                chunks.append({
                    "text": split,
                    "metadata": {
                        "source":         metadata["filename"],
                        "title":          metadata["title"],
                        "authors":        metadata["authors"],
                        "year":           metadata["year"],
                        "doi":            metadata.get("doi"),
                        "section":        section_name,
                        "section_priority": priority,
                        "chunk_index":    i,
                        "chunk_type":     chunk_type,
                        "char_count":     len(split),
                        "word_count":     len(split.split()),
                    }
                })
    else:
        # fallback: chunk by page
        for page in paper["pages"]:
            if not page["text"].strip():
                continue
            splits = splitter.split_text(page["text"])
            for i, split in enumerate(splits):
                if len(split.strip()) < 40:
                    continue
                chunks.append({
                    "text": split,
                    "metadata": {
                        "source":           metadata["filename"],
                        "title":            metadata["title"],
                        "authors":          metadata["authors"],
                        "year":             metadata["year"],
                        "doi":              metadata.get("doi"),
                        "section":          f"page_{page['page']}",
                        "section_priority": 0.5,
                        "chunk_index":      i,
                        "chunk_type":       detect_chunk_type(split),
                        "char_count":       len(split),
                        "word_count":       len(split.split()),
                    }
                })

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

        print(f"  {paper['metadata']['filename'][:45]}: "
              f"{len(paper_chunks)} chunks | types: {types}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: list[dict],
                output_path: str = "indexes/chunks_metadata.json"):
    Path("indexes").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Chunks saved → {output_path}")


if __name__ == "__main__":
    from src.ingest import load_all_papers
    papers = load_all_papers("data/papers")
    chunks = chunk_all_papers(papers)
    save_chunks(chunks)
    print("\nSample chunk:")
    print(chunks[0]["text"][:300])
    print("Metadata:", chunks[0]["metadata"])