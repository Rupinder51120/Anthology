"""
Document parser using Docling.
Extracts: text blocks, tables, figures, captions, equations.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#Minimum content thresholds

_MIN_TEXT_LEN      = 30    # was 20 — 20 chars is often a header fragment or noise
_MIN_TABLE_ROWS    = 2     # skip single-row "tables" Docling mis-classifies
_MIN_PARA_LEN      = 40    # PyMuPDF fallback paragraph threshold

# Section header pattern for PyMuPDF fallback
# Matches: "3. Results", "4.2 Discussion", "CONCLUSION", "RELATED WORK"
_SECTION_RE = re.compile(
    r"^\d+\.?\d*\s+[A-Z]|^[A-Z][A-Z\s]{3,}$"
)


#Data contract

@dataclass
class ParsedBlock:
    content_type:   str             # text | table | figure | equation | caption
    content:        str             # primary text content
    page_number:    int
    section_title:  str
    figure_number:  str | None = None
    image_path:     str | None = None
    table_markdown: str | None = None
    metadata:       dict = field(default_factory=dict)


#Public entry point

def parse_pdf(
    pdf_path: str,
    figures_dir: str = "data/figures",
) -> list[ParsedBlock]:
    """
    Parse a PDF and return a list of ParsedBlock objects.
    Tries Docling first; falls back to PyMuPDF on ImportError or any failure.
    Always returns a list — never raises.
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return []
    if path.stat().st_size == 0:
        logger.error("PDF is empty: %s", pdf_path)
        return []

    try:
        blocks = _parse_with_docling(pdf_path, figures_dir)
        if not blocks:
            # Docling succeeded but extracted nothing — treat as failure
            logger.warning(
                "Docling returned 0 blocks for %s, falling back to PyMuPDF",
                pdf_path,
            )
            return _parse_with_pymupdf(pdf_path)
        return blocks

    except ImportError:
        logger.warning("Docling not installed — falling back to PyMuPDF")
        return _parse_with_pymupdf(pdf_path)

    except Exception as e:
        logger.warning("Docling failed for %s (%s) — falling back to PyMuPDF", pdf_path, e)
        return _parse_with_pymupdf(pdf_path)


#Docling backend 

def _parse_with_docling(pdf_path: str, figures_dir: str) -> list[ParsedBlock]:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr                  = False
    pipeline_options.do_table_structure      = True
    pipeline_options.images_scale            = 2.0
    pipeline_options.generate_page_images    = False
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path)
    doc    = result.document

    blocks:          list[ParsedBlock] = []
    current_section: str               = "preamble"
    figure_counter:  int               = 0
    table_counter:   int               = 0

    try:
        for element, _ in doc.iterate_items():
            element_type = type(element).__name__

            #section header 
            if element_type == "SectionHeaderItem":
                text = (element.text or "").strip()
                if text:
                    current_section = text

            #text block 
            elif element_type == "TextItem":
                text = (element.text or "").strip()
                if len(text) < _MIN_TEXT_LEN:
                    continue
                blocks.append(ParsedBlock(
                    content_type  = "text",
                    content       = text,
                    page_number   = _page_no(element),
                    section_title = current_section,
                ))

            # table 
            elif element_type == "TableItem":
                md = element.export_to_markdown(doc) or ""
                md = md.strip()

                # skip degenerate tables (single row or nearly empty)
                row_count = md.count("\n")
                if not md or row_count < _MIN_TABLE_ROWS:
                    continue

                table_counter += 1
                blocks.append(ParsedBlock(
                    content_type   = "table",
                    content        = md,
                    page_number    = _page_no(element),
                    section_title  = current_section,
                    figure_number  = f"Table {table_counter}",
                    table_markdown = md,
                ))

            # ── figure ────────────────────────────────────────────────────────
            elif element_type == "PictureItem":
                figure_counter += 1
                saved_path = _save_figure(element, doc, pdf_path, figure_counter, figures_path)
                caption    = _extract_caption(element, figure_counter, _page_no(element))

                blocks.append(ParsedBlock(
                    content_type  = "figure",
                    content       = caption,
                    page_number   = _page_no(element),
                    section_title = current_section,
                    figure_number = f"Figure {figure_counter}",
                    image_path    = saved_path,
                ))

    finally:
        # Release Docling's document object explicitly so GC can reclaim
        # the full in-memory representation before we return.
        del doc, result
        gc.collect()

    return blocks


# ── Docling helpers ───────────────────────────────────────────────────────────

def _page_no(element) -> int:
    """Safely extract page number from a Docling element."""
    try:
        return element.prov[0].page_no if element.prov else 0
    except (IndexError, AttributeError):
        return 0


def _save_figure(
    element,
    doc,
    pdf_path:       str,
    figure_counter: int,
    figures_path:   Path,
) -> str | None:
    """
    Save a PictureItem to disk.
    Returns the absolute path string on success, None on failure.
    Ensures the PIL Image is always closed to free pixel buffer memory.
    """
    fig_id       = hashlib.md5(f"{pdf_path}_{figure_counter}".encode()).hexdigest()[:8]
    img_filename = f"{fig_id}.png"
    img_path     = figures_path / img_filename

    # Already saved from a previous (interrupted) run — skip re-encode
    if img_path.exists():
        return str(img_path)

    img = None
    try:
        img = element.get_image(doc)
        if img is None:
            return None
        img.save(str(img_path))
        return str(img_path)
    except Exception as e:
        logger.warning("Could not save figure %d: %s", figure_counter, e)
        return None
    finally:
        # PIL Image pixel buffer freed immediately regardless of success/failure
        if img is not None:
            try:
                img.close()
            except Exception:
                pass


def _extract_caption(element, figure_counter: int, page_no: int) -> str:
    """
    Extract caption text from a PictureItem.
    Falls back to a descriptive placeholder — never returns empty string
    so the chunk always has embeddable content.
    """
    # Docling stores caption as a CaptionItem child or direct attribute
    caption = ""

    # Try direct attribute first
    if hasattr(element, "caption") and element.caption:
        caption = str(element.caption).strip()

    # Try captions list (newer Docling versions)
    if not caption and hasattr(element, "captions"):
        for cap in (element.captions or []):
            text = getattr(cap, "text", None) or str(cap)
            if text and text.strip():
                caption = text.strip()
                break

    if not caption:
        caption = f"Figure {figure_counter} on page {page_no}"

    return caption


# ── PyMuPDF fallback ──────────────────────────────────────────────────────────

def _parse_with_pymupdf(pdf_path: str) -> list[ParsedBlock]:
    """
    Dumb text extraction via PyMuPDF.
    No figure/table extraction — text only.
    Used only when Docling is unavailable or fails entirely.
    """
    try:
        import fitz
    except ImportError:
        logger.error("Neither Docling nor PyMuPDF (fitz) is installed.")
        return []

    doc             = fitz.open(pdf_path)
    blocks:         list[ParsedBlock] = []
    current_section: str              = "preamble"

    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                continue

            # Split on double newline (paragraph boundary)
            raw_paras = text.split("\n\n")

            for raw in raw_paras:
                para = raw.strip()
                if not para:
                    continue

                # Detect section headers
                if _SECTION_RE.match(para) and len(para) < 120:
                    current_section = para[:100]
                    continue

                # Skip noise: page numbers, single words, very short lines
                if len(para) < _MIN_PARA_LEN:
                    continue

                # Skip lines that are only digits (page numbers, equation numbers)
                if re.fullmatch(r"[\d\s\.\-]+", para):
                    continue

                blocks.append(ParsedBlock(
                    content_type  = "text",
                    content       = _clean_pymupdf_text(para),
                    page_number   = page_num,
                    section_title = current_section,
                ))
    finally:
        doc.close()

    return blocks


def _clean_pymupdf_text(text: str) -> str:
    """
    Clean common PyMuPDF extraction artifacts.
    - Collapse mid-word hyphen line breaks: "atten-\ntion" → "attention"
    - Collapse single newlines within a paragraph to spaces
    - Normalize whitespace
    """
    # Rejoin hyphenated line breaks
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse remaining single newlines to space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()