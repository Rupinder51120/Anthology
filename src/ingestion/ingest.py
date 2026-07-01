

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Section headers used by extract_sections() ────────────────────────────────

SECTION_HEADERS: list[str] = [
    "abstract", "introduction", "background", "related work",
    "literature review", "preliminary", "preliminaries",
    "methodology", "method", "methods", "approach", "proposed method",
    "architecture", "model", "framework",
    "experiment", "experiments", "experimental setup", "experimental results",
    "results", "evaluation", "analysis", "discussion",
    "conclusion", "conclusions", "future work", "references", "appendix",
]


# ── Registry loader ───────────────────────────────────────────────────────────

_registry_cache: dict | None = None


def load_registry(path: str = "data/download_registry.json") -> dict:
    global _registry_cache
    if _registry_cache is None:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            # Re-index by filename for O(1) lookup
            _registry_cache = {
                v["filename"]: v
                for v in data.values()
                if "filename" in v
            }
        else:
            _registry_cache = {}
    return _registry_cache


def extract_metadata_from_registry(filename: str) -> dict | None:
    """
    Primary metadata source — uses download_registry.json.
    Returns a fully-populated metadata dict or None when filename is absent.
    """
    entry = load_registry().get(filename)
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
        "title":     entry.get("title", filename.replace(".pdf", "")),
        "authors":   authors_str,
        "year":      str(entry.get("year", "Unknown")),
        "doi":       entry.get("doi"),
        "arxiv_id":  entry.get("arxiv_id"),
        "arxiv_url": entry.get("url"),
        "topic":     entry.get("topic", ""),
        "abstract":  (entry.get("abstract", "") or "")[:300],
        "filename":  filename,
    }


# ── Compiled patterns ─────────────────────────────────────────────────────────

_TITLE_HARD_REJECT = re.compile(
    r"""
    (?:^|\s)(figure|fig\.)\s+\d+        # figure captions
    | (?:^|\s)table\s+\d+               # table captions
    | ^doi:\s*10\.                       # explicit DOI line
    | \b10\.\d{4,9}/\S{5,}              # inline DOI
    | https?://                          # URLs
    | @\S+\.\S+                          # email addresses
    | \©|\(c\)\s*20\d{2}               # copyright symbols
    | all\s+rights\s+reserved
    | under\s+(the\s+)?(cc|creative\s+commons|mit\s+license|apache|gpl)\b
    | published\s+by\s+
    | ieee\s+(transactions|access|conference|letters|journal)\b
    | \bproceedings\s+of\s+(the\s+)?
    | \barxiv:\d{4}\.\d+
    | \bpreprint\b.*\bunder\s+review\b
    | \bsubmitted\s+to\b
    | \b(received|revised|accepted)\s+\w+\s+\d{4}
    | \bvol\.\s*\d+\b
    | \bno\.\s*\d+\b
    | \bpp\.\s*\d+
    | \bissn\b|\bisbn\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_AFFILIATION_MARKERS = re.compile(
    r"""
    \b(university|universit[äéy]|université|universidad
      |institute|institution
      |department|dept\.?
      |laboratory|laboratoire
      |school\s+of|faculty\s+of|college\s+of
      |center\s+for|centre\s+for
      |division\s+of
      |research\s+group
      )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NAME_TOKEN = re.compile(
    r"[A-ZÁÉÍÓÚÀÈÙÄÖÜÂÊÎÔÛÑ][a-záéíóúàèùäöüâêîôûñ\-]+"
    r"(?:\s+[A-Z]\.)*"
    r"\s+[A-ZÁÉÍÓÚÀÈÙÄÖÜÂÊÎÔÛÑ][a-záéíóúàèùäöüâêîôûñ\-]+"
)

_BODY_STARTERS = re.compile(
    r"^(abstract|keywords?|introduction|1[\.\s]+introduction"
    r"|we\s+(present|propose|show|survey|study|introduce))",
    re.IGNORECASE,
)

_AFFIL_SKIP = re.compile(
    r"""
    university|institute|department|dept\.|laboratory|lab\b
    |school\s+of|faculty|college|center\s+for|centre\s+for|division
    |@\S+\.\S+
    |https?://
    |^abstract[\s:]|^keywords?[\s:]
    |^(we\s+|this\s+(paper|work)|in\s+this)
    |©|\(c\)\s*20\d{2}
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _strip_superscripts(line: str) -> str:
    return re.sub(r'(?<=[a-zA-Z])\s*[\d\*†‡§¶]+(?=[,;\s]|$)', '', line)


def _count_name_tokens(line: str) -> int:
    return len(_NAME_TOKEN.findall(_strip_superscripts(line)))


def _is_author_list(line: str) -> bool:
    cleaned = _strip_superscripts(line)
    parts   = [p.strip() for p in re.split(r'[,;]|\band\b', cleaned) if p.strip()]
    return sum(1 for p in parts if _NAME_TOKEN.search(p)) >= 2


def score_title_candidate(line: str, position: int) -> float:
    """
    Score a candidate line for likelihood of being the paper title.
    Returns float('-inf') for definitive non-titles.
    Higher score → more likely to be the title.

    Public (no underscore) — metadata_resolver.py imports this directly
    to re-validate Docling's TitleItem guess against the same hard-reject
    rules applied to every other candidate line.
    """
    if _TITLE_HARD_REJECT.search(line):
        return float('-inf')
    if _BODY_STARTERS.match(line):
        return float('-inf')
    if _AFFILIATION_MARKERS.search(line) and _count_name_tokens(line) < 2:
        return float('-inf')
    if _is_author_list(line):
        return float('-inf')

    score      = 0.0
    L          = len(line)
    words      = line.split()
    word_count = len(words)

    # Length
    if L < 15:
        score -= 6.0
    elif 15 <= L <= 200:
        score += 4.0
    elif 200 < L <= 350:
        score += 0.0
    else:
        score -= 12.0

    # Word count
    if word_count < 3:
        score -= 4.0
    elif word_count >= 5:
        score += 1.5

    # Comma penalty
    comma_count = line.count(',')
    if comma_count >= 4:
        score -= 4.0
    elif comma_count >= 2:
        score -= 1.5

    # Affiliation boilerplate
    if _AFFILIATION_MARKERS.search(line):
        score -= 6.0

    # Isolated trailing digits (affiliation markers on non-author lines)
    isolated = len(re.findall(r'(?<=[a-zA-Z])\s*\d+(?=[,;\s]|$)', line))
    if isolated >= 2:
        score -= 4.0

    # Title-case fraction
    if word_count >= 4:
        tc   = sum(1 for w in words if w and w[0].isupper())
        frac = tc / word_count
        if frac >= 0.55:
            score += 2.5
        elif frac <= 0.2:
            score -= 2.0

    # Structural cues
    if ':' in line:
        score += 1.0
    if line.endswith('.'):
        score -= 2.0
    if line.endswith('?'):
        score += 0.5

    # Position bonus
    if position == 0:
        score += 1.0
    elif position <= 3:
        score += 0.5

    return score


# ── Author extraction ─────────────────────────────────────────────────────────

def _extract_authors(lines: list[str], title: str) -> str:
    title_idx = -1
    for i, line in enumerate(lines[:30]):
        if line == title or (len(title) >= 20 and line.startswith(title[:60])):
            title_idx = i
            break

    start = title_idx + 1 if title_idx >= 0 else 1
    end   = start + 20

    for line in lines[start:end]:
        if len(line) < 4 or len(line) > 400:
            continue
        if _AFFIL_SKIP.search(line):
            continue
        if _BODY_STARTERS.match(line):
            break
        if _is_author_list(line):
            cleaned = _strip_superscripts(line)
            cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip(' ,;')
            if cleaned:
                return cleaned

    return "Unknown"


# ── Heuristic PDF metadata extractor ─────────────────────────────────────────

def extract_metadata_from_pdf(first_page_text: str, filename: str) -> dict:
    """
    Fallback metadata extractor for papers not in the registry.

    Receives first_page_text assembled by ParsedDocument (from Docling elements
    on page 1) and applies scoring-based title selection across the first 30 lines.

    Defensive truncation (title[:450], authors[:1000]) guarantees no DB overflow.
    """
    lines        = [ln.strip() for ln in first_page_text.split("\n") if ln.strip()]
    safe_fallback = filename.replace(".pdf", "").replace("_", " ")
    title         = safe_fallback

    if lines:
        scored = [
            (score_title_candidate(ln, i), i, ln)
            for i, ln in enumerate(lines[:30])
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        best_score, _, best_line = scored[0]
        if best_score > float('-inf') and len(best_line) >= 10:
            title = best_line

    author_line = _extract_authors(lines, title)
    year_match  = re.search(r"\b(20\d{2})\b", first_page_text)
    year        = year_match.group() if year_match else "Unknown"
    doi_match   = re.search(r"10\.\d{4,9}/[^\s\]\)]+", first_page_text)
    doi         = doi_match.group() if doi_match else None

    return {
        "title":     title[:450],
        "authors":   author_line[:1000],
        "year":      year,
        "doi":       doi,
        "arxiv_id":  None,
        "arxiv_url": None,
        "topic":     "",
        "abstract":  "",
        "filename":  filename,
    }


# ── Section extractor ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'http[s]?://(?!doi)\S+', '[URL]', text)
    return text.strip()


def extract_sections(full_text: str) -> dict:
    """
    Split plain text into a {section_name: text} dict by matching section headers.
    Used by metadata_resolver.py for abstract backfill when Docling finds no
    dedicated AbstractItem.
    """
    sections:        dict[str, str] = {}
    current_section: str            = "preamble"
    current_text:    list[str]      = []

    for line in full_text.split("\n"):
        stripped = line.strip().lower()
        matched  = None

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