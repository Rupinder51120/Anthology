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


def clean_text(text: str) -> str:
    # remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # remove page numbers standing alone
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # remove URLs (keep DOIs)
    text = re.sub(r'http[s]?://(?!doi)\S+', '[URL]', text)
    return text.strip()


def extract_metadata_from_first_page(first_page_text: str, filename: str) -> dict:
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

    # title: longest line in first 5 non-empty lines (avoids journal headers)
    candidates = lines[:8]
    title = max(candidates, key=len) if candidates else filename.replace(".pdf", "")
    if len(title) < 10:
        title = filename.replace(".pdf", "").replace("_", " ")

    # year
    year_match = re.search(r"\b(19|20)\d{2}\b", first_page_text)
    year = year_match.group() if year_match else "Unknown"

    # DOI
    doi_match = re.search(r"10\.\d{4,9}/[^\s\]]+", first_page_text)
    doi = doi_match.group() if doi_match else None

    # authors: lines after title with no digits, not institution
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

    # abstract: look for it explicitly
    abstract = ""
    text_lower = first_page_text.lower()
    abs_idx = text_lower.find("abstract")
    if abs_idx != -1:
        abstract = first_page_text[abs_idx + 8:abs_idx + 600].strip()
        abstract = re.sub(r'\n', ' ', abstract)

    return {
        "title": title,
        "authors": author_line,
        "year": year,
        "doi": doi,
        "filename": filename,
        "abstract_snippet": abstract[:300] if abstract else ""
    }


def extract_sections(full_text: str) -> dict:
    sections = {}
    current_section = "preamble"
    current_text = []

    for line in full_text.split("\n"):
        stripped = line.strip().lower()

        # match section headers — exact or starts with header + space/number
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
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        sections[current_section] = clean_text("\n".join(current_text))

    return sections


def extract_figures_tables(full_text: str) -> list[dict]:
    """Extract figure and table captions — useful for context."""
    captions = []
    pattern = re.compile(
        r'(Figure|Fig\.|Table)\s+(\d+)[:\.]?\s+([^\n]{10,200})',
        re.IGNORECASE
    )
    for match in pattern.finditer(full_text):
        captions.append({
            "type": match.group(1),
            "number": match.group(2),
            "caption": match.group(3).strip()
        })
    return captions


def load_paper(pdf_path: Path) -> dict:
    doc = fitz.open(str(pdf_path))
    filename = pdf_path.name
    pages = []
    full_text = ""

    for page_num, page in enumerate(doc, start=1):
        # use "blocks" mode for better column handling
        blocks = page.get_text("blocks")
        page_text = "\n".join(
            b[4] for b in sorted(blocks, key=lambda b: (b[1], b[0]))
            if isinstance(b[4], str)
        )
        pages.append({
            "page": page_num,
            "text": page_text,
            "char_count": len(page_text)
        })
        full_text += page_text + "\n"

    doc.close()

    metadata   = extract_metadata_from_first_page(pages[0]["text"], filename)
    sections   = extract_sections(full_text)
    captions   = extract_figures_tables(full_text)

    # word count stats
    word_count = len(full_text.split())
    page_count = len(pages)

    return {
        "metadata": metadata,
        "pages": pages,
        "sections": sections,
        "captions": captions,
        "full_text": full_text,
        "stats": {
            "pages": page_count,
            "words": word_count,
            "sections_found": list(sections.keys())
        }
    }


def load_all_papers(papers_dir: str = "data/papers") -> list[dict]:
    papers_path = Path(papers_dir)
    pdf_files   = sorted(papers_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {papers_dir}")
        return []

    all_papers = []
    failed     = []

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Loading: {pdf_file.name[:55]}")
        try:
            paper = load_paper(pdf_file)
            all_papers.append(paper)
            print(f"  Pages: {paper['stats']['pages']} | "
                  f"Words: {paper['stats']['words']} | "
                  f"Sections: {paper['stats']['sections_found']}")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(pdf_file.name)

    print(f"\nLoaded: {len(all_papers)} | Failed: {len(failed)}")
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


if __name__ == "__main__":
    papers = load_all_papers("data/papers")
    save_metadata(papers)
    if papers:
        p = papers[0]
        print("\n--- Sample ---")
        print("Title:",    p["metadata"]["title"])
        print("Authors:",  p["metadata"]["authors"])
        print("Year:",     p["metadata"]["year"])
        print("Sections:", p["stats"]["sections_found"])
        print("Captions:", len(p["captions"]))