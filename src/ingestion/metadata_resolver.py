
from __future__ import annotations

import logging

from src.ingestion.ingest import (
    extract_metadata_from_registry,
    extract_metadata_from_pdf,
    extract_sections,
    score_title_candidate,
)
from src.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


def resolve_metadata(parsed_doc: ParsedDocument, filename: str) -> dict:
    """
    Returns the resolved metadata dict for one paper, plus a "metadata_source"
    field documenting which path won.

    Resolution Order:
    1. Registry (Strongest)
    2. Docling Structural Metadata (Strong)
    3. Heuristic Extraction (Fallback)
    """
    # ── 1. Registry Check ──────────────────────────────────────────────────────
    registry_meta = extract_metadata_from_registry(filename)
    if registry_meta:
        logger.info("Metadata for %s resolved from registry", filename)
        return {**registry_meta, "metadata_source": "registry"}

    # ── 2. Docling Structural Metadata ─────────────────────────────────────────
    docling_meta = parsed_doc.metadata
    if docling_meta:
        resolved = {}
        # Map Docling fields to internal schema
        if "title" in docling_meta and docling_meta["title"]:
            resolved["title"] = str(docling_meta["title"])

        if "authors" in docling_meta and docling_meta["authors"]:
            authors = docling_meta["authors"]
            resolved["authors"] = ", ".join(map(str, authors)) if isinstance(authors, list) else str(authors)

        if "date" in docling_meta and docling_meta["date"]:
            date_str = str(docling_meta["date"])
            year_match = re.search(r"\b(20\d{2}|19\d{2})\b", date_str)
            if year_match:
                resolved["year"] = year_match.group()

        if "doi" in docling_meta and docling_meta["doi"]:
            resolved["doi"] = str(docling_meta["doi"])

        if "arxiv_id" in docling_meta and docling_meta["arxiv_id"]:
            resolved["arxiv_id"] = str(docling_meta["arxiv_id"])

        if "url" in docling_meta and docling_meta["url"]:
            resolved["url"] = str(docling_meta["url"])

        if resolved.get("title") or resolved.get("authors"):
            logger.info("Metadata for %s resolved from Docling structural metadata", filename)
            heuristic_meta = extract_metadata_from_pdf(parsed_doc.first_page_text, filename)
            final_meta = {**heuristic_meta, **resolved}

            docling_title = parsed_doc.doc_title_guess
            if docling_title:
                score = score_title_candidate(docling_title, 0)
                if score > float("-inf") and len(docling_title) >= 10:
                    final_meta["title"] = docling_title[:450]

            if not final_meta.get("abstract"):
                abstract_text = ""
                for block in parsed_doc.blocks:
                    if block.section_title and block.section_title.lower() == "abstract":
                        abstract_text += block.content + "\n"
                        if len(abstract_text) > 500:
                            break
                if abstract_text.strip():
                    final_meta["abstract"] = abstract_text.strip()[:300]
                else:
                    try:
                        sections = extract_sections(parsed_doc.first_page_text)
                        if sections.get("abstract"):
                            final_meta["abstract"] = sections["abstract"][:300]
                    except Exception as e:
                        logger.warning("Abstract backfill failed for %s: %s", filename, e)

            final_meta["metadata_source"] = "docling_structural"
            return final_meta

    # ── 3. Heuristic Extraction (Final Fallback) ───────────────────────────────
    heuristic_meta = extract_metadata_from_pdf(parsed_doc.first_page_text, filename)

    # Refine title with Docling guess
    docling_title = parsed_doc.doc_title_guess
    used_docling_title = False
    if docling_title:
        score = score_title_candidate(docling_title, 0)
        if score > float("-inf") and len(docling_title) >= 10:
            heuristic_meta["title"] = docling_title[:450]
            used_docling_title = True

    # Abstract: heuristic extractor never populates this field on on its own.
    # Prefer structured blocks where section_title is 'abstract'.
    if not heuristic_meta.get("abstract"):
        abstract_text = ""
        for block in parsed_doc.blocks:
            if block.section_title and block.section_title.lower() == "abstract":
                abstract_text += block.content + "\n"
                if len(abstract_text) > 500:
                    break

        if abstract_text.strip():
            heuristic_meta["abstract"] = abstract_text.strip()[:300]
        else:
            # Fallback to the plain-text section-header matcher
            try:
                sections = extract_sections(parsed_doc.first_page_text)
                if sections.get("abstract"):
                    heuristic_meta["abstract"] = sections["abstract"][:300]
            except Exception as e:
                logger.warning("Abstract backfill failed for %s: %s", filename, e)

    heuristic_meta["metadata_source"] = (
        "docling_title+heuristic" if used_docling_title else "heuristic"
    )
    return heuristic_meta