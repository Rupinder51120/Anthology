import fitz
import json
import re
from pathlib import Path


SECTION_HEADERS = [
    "abstract", "introduction", "background", "related work",
    "literature review", "preliminary", "preliminaries",
    "methodology", "method", "methods", "approach", "proposed method",
    "architecture", "model", "framework",
    "experiment", "experiments", "experimental setup", "experimental results",
    "results", "evaluation", "analysis", "discussion",
    "conclusion", "conclusions", "future work", "references", "appendix"
]

# ─── registry loader ──────────────────────────────────────────

_registry_cache = None

def load_registry(path: str = "data/download_registry.json") -> dict:
    global _registry_cache
    if _registry_cache is None:
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            # reindex by filename for fast lookup
            _registry_cache = {
                v["filename"]: v
                for v in data.values()
                if "filename" in v
            }
        else:
            _registry_cache = {}
    return _registry_cache


def get_metadata_from_registry(filename: str) -> dict | None:
    registry = load_registry()
    return registry.get(filename)


# ─── text cleaning ────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'http[s]?://(?!doi)\S+', '[URL]', text)
    return text.strip()


# ─── metadata extraction ──────────────────────────────────────

def extract_metadata_from_registry(filename: str) -> dict | None:
    """
    Primary metadata source — uses download_registry.json.
    Has correct title, authors, year, arxiv_id for every downloaded paper.
    """
    entry = get_metadata_from_registry(filename)
    if not entry:
        return None

    authors = entry.get("authors", [])
    if isinstance(authors, list):
        authors_str = ", ".join(authors[:3])
        if len(authors) > 3:
            authors_str += " et al."
    else:
        authors_str = str(authors)

    return {
        "title":           entry.get("title", filename.replace(".pdf", "")),
        "authors":         authors_str,
        "year":            str(entry.get("year", "Unknown")),
        "doi":             entry.get("doi"),
        "arxiv_id":        entry.get("arxiv_id"),
        "arxiv_url":       entry.get("url"),
        "topic":           entry.get("topic", ""),
        "abstract":        entry.get("abstract", "")[:300],
        "filename":        filename,
    }


def extract_metadata_from_pdf(first_page_text: str, filename: str) -> dict:
    """
    Fallback metadata — parses PDF text when registry entry missing.
    Less reliable but works for manually added papers.
    """
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

    # title: longest line in first 8
    candidates = lines[:8]
    title = max(candidates, key=len) if candidates else filename.replace(".pdf", "")
    if len(title) < 10:
        title = filename.replace(".pdf", "").replace("_", " ")

    # year
    year_match = re.search(r"\b(20\d{2})\b", first_page_text)
    year = year_match.group() if year_match else "Unknown"

    # DOI
    doi_match = re.search(r"10\.\d{4,9}/[^\s\]]+", first_page_text)
    doi = doi_match.group() if doi_match else None

    # authors
    author_line = "Unknown"
    for line in lines[1:8]:
        if len(line) < 5:
            continue
        if any(w in line.lower() for w in ["university", "institute", "dept",
                                            "@", "lab", "abstract", "http",
                                            "journal", "conference", "arxiv"]):
            continue
        if not any(char.isdigit() for char in line):
            author_line = line
            break

    return {
        "title":     title,
        "authors":   author_line,
        "year":      year,
        "doi":       doi,
        "arxiv_id":  None,
        "arxiv_url": None,
        "topic":     "",
        "abstract":  "",
        "filename":  filename,
    }


# ─── section extraction ───────────────────────────────────────

def extract_sections(full_text: str) -> dict:
    sections = {}
    current_section = "preamble"
    current_text    = []

    for line in full_text.split("\n"):
        stripped = line.strip().lower()

        matched = None
        for h in SECTION_HEADERS:
            if (stripped == h
                    or stripped.startswith(h + " ")
                    or stripped.startswith(h + ".")
                    or re.match(rf'^\d+\.?\s+{re.escape(h)}', stripped)):
                matched = h
                break

        if matched:
            if current_text:
                sections[current_section] = clean_text("\n".join(current_text))
            current_section = matched
            current_text    = []
        else:
            current_text.append(line)

    if current_text:
        sections[current_section] = clean_text("\n".join(current_text))

    return sections


def extract_figures_tables(full_text: str) -> list[dict]:
    captions = []
    pattern  = re.compile(
        r'(Figure|Fig\.|Table)\s+(\d+)[:\.]?\s+([^\n]{10,200})',
        re.IGNORECASE
    )
    for match in pattern.finditer(full_text):
        captions.append({
            "type":    match.group(1),
            "number":  match.group(2),
            "caption": match.group(3).strip()
        })
    return captions


# ─── paper loader ─────────────────────────────────────────────

def load_paper(pdf_path: Path) -> dict:
    doc      = fitz.open(str(pdf_path))
    filename = pdf_path.name
    pages    = []
    full_text = ""

    for page_num, page in enumerate(doc, start=1):
        blocks    = page.get_text("blocks")
        page_text = "\n".join(
            b[4] for b in sorted(blocks, key=lambda b: (b[1], b[0]))
            if isinstance(b[4], str)
        )
        pages.append({
            "page":       page_num,
            "text":       page_text,
            "char_count": len(page_text)
        })
        full_text += page_text + "\n"

    doc.close()

    # ── metadata: registry first, PDF fallback ──
    metadata = extract_metadata_from_registry(filename)
    if metadata:
        source = "registry"
    else:
        print(f"  Warning: {filename} not in registry, using PDF extraction")
        metadata = extract_metadata_from_pdf(pages[0]["text"], filename)
        source   = "pdf"

    sections = extract_sections(full_text)
    captions = extract_figures_tables(full_text)

    # inject registry abstract into sections if missing
    if "abstract" not in sections and metadata.get("abstract"):
        sections["abstract"] = metadata["abstract"]

    return {
        "metadata":      metadata,
        "metadata_source": source,
        "pages":         pages,
        "sections":      sections,
        "captions":      captions,
        "full_text":     full_text,
        "stats": {
            "pages":          len(pages),
            "words":          len(full_text.split()),
            "sections_found": list(sections.keys())
        }
    }


# ─── batch loader ─────────────────────────────────────────────

def load_all_papers(papers_dir: str = "data/papers") -> list[dict]:
    papers_path = Path(papers_dir)
    pdf_files   = sorted(papers_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {papers_dir}")
        return []

    registry    = load_registry()
    all_papers  = []
    failed      = []
    from_registry = 0
    from_pdf      = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_file.name[:55]}")
        try:
            paper = load_paper(pdf_file)
            all_papers.append(paper)

            if paper["metadata_source"] == "registry":
                from_registry += 1
                print(f"  ✓ [{paper['metadata_source']}] "
                      f"{paper['metadata']['title'][:50]} "
                      f"({paper['metadata']['year']})")
            else:
                from_pdf += 1
                print(f"  ⚠ [{paper['metadata_source']}] "
                      f"{paper['metadata']['title'][:50]}")

        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(pdf_file.name)

    print(f"\n{'─'*50}")
    print(f"Loaded:        {len(all_papers)} papers")
    print(f"From registry: {from_registry} (clean metadata)")
    print(f"From PDF:      {from_pdf} (fallback)")
    print(f"Failed:        {len(failed)}")
    if failed:
        print(f"Failed files: {failed}")

    return all_papers


def save_metadata(all_papers: list[dict],
                  output_path: str = "indexes/papers_metadata.json"):
    Path("indexes").mkdir(exist_ok=True)
    metadata_list = [p["metadata"] for p in all_papers]
    with open(output_path, "w") as f:
        json.dump(metadata_list, f, indent=2)
    print(f"Metadata saved → {output_path} ({len(metadata_list)} papers)")


# ─── test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    papers = load_all_papers("data/papers")
    save_metadata(papers)

    if papers:
        p = papers[0]
        print("\n--- Sample ---")
        print(f"Title:   {p['metadata']['title']}")
        print(f"Authors: {p['metadata']['authors']}")
        print(f"Year:    {p['metadata']['year']}")
        print(f"Source:  {p['metadata_source']}")
        print(f"Topic:   {p['metadata']['topic']}")
        print(f"Arxiv:   {p['metadata'].get('arxiv_url')}")