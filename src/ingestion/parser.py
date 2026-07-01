from __future__ import annotations
import time
import gc
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Thresholds ────────────────────────────────────────────────────────────────

_MIN_TEXT_LEN   = 30   # chars — shorter blocks are noise or lone headers
_MIN_TABLE_ROWS = 2    # Docling sometimes emits 1-row "tables" for borders
_MIN_PARA_LEN   = 40   # used for caption fallback cleaning


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    """
    Single semantically-typed content unit extracted from a PDF.

    Fields
    ------
    content_type   : "text" | "table" | "figure" | "equation" | "caption"
    content        : primary embeddable text (caption, cell text, equation LaTeX)
    page_number    : 1-indexed; 0 means unknown
    section_title  : enclosing section header at extraction time
    figure_number  : "Figure 3" / "Table 2" — None for plain text/equation
    image_path     : absolute path to saved PNG — figures only
    table_markdown : raw Docling markdown export — tables only
    is_enriched    : set True by ingest_service after captioning/summarisation
    metadata       : spare dict for downstream annotation
    """
    content_type:   str
    content:        str
    page_number:    int
    section_title:  str
    figure_number:  str | None = None
    image_path:     str | None = None
    table_markdown: str | None = None
    is_enriched:    bool       = False
    metadata:       dict       = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """
    Complete structured representation of one parsed PDF.

    Fields
    ------
    blocks          : all extracted blocks in document reading order
    first_page_text : plain-text join of page-1 content — for metadata heuristics
    doc_title_guess : Docling's TitleItem text, or None when absent/PyMuPDF path
    metadata        : structural metadata extracted by Docling (title, authors, etc.)
    section_map     : section_title → [indices into blocks] for fast section lookup
    """
    blocks:          list[ParsedBlock]
    first_page_text: str
    doc_title_guess: str | None
    metadata:        dict = field(default_factory=dict)
    section_map:     dict[str, list[int]] = field(default_factory=dict)


# ── Public entry point ────────────────────────────────────────────────────────

def parse_pdf_document(
    pdf_path:    str,
    figures_dir: str = "data/figures",
) -> ParsedDocument:
    """
    Parse a PDF with Docling and return a ParsedDocument.

    Always returns a valid (possibly empty) ParsedDocument — never raises.
    The caller (ingest_service.py) should check len(doc.blocks) == 0 and
    treat that as an ingestion failure.
    """
    path = Path(pdf_path)

    if not path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return _empty_doc()
    if path.stat().st_size == 0:
        logger.error("PDF is empty: %s", pdf_path)
        return _empty_doc()

    try:
        return _parse_with_docling(str(path), figures_dir)
    except Exception as e:
        logger.error("Docling parse failed for %s: %s", pdf_path, e, exc_info=True)
        return _empty_doc()


# ── Docling backend ───────────────────────────────────────────────────────────

def _parse_with_docling(pdf_path: str, figures_dir: str) -> ParsedDocument:
    parse_start = time.perf_counter()

    logger.info("Starting Docling parser...")

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr                  = True # Enable OCR for image-only PDFs
    pipeline_options.do_table_structure      = True
    pipeline_options.images_scale            = 2.0
    pipeline_options.generate_page_images    = False
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Initialise state outside try block
    result = None
    doc = None
    blocks:          list[ParsedBlock] = []
    current_section: str               = "preamble"
    figure_counter:  int               = 0
    table_counter:   int               = 0
    doc_title_guess: str | None        = None
    first_page_lines: list[str]        = []
    doc_metadata:    dict              = {}

    try:
        convert_start = time.perf_counter()
        result = converter.convert(pdf_path)
        doc    = result.document
        logger.info(
            "Docling convert() took %.2fs",
            time.perf_counter() - convert_start
        )

        # ── Metadata Extraction (must happen while doc exists) ───────────────────
        if hasattr(doc, "metadata") and doc.metadata:
            meta_obj = doc.metadata
            try:
                if isinstance(meta_obj, dict):
                    doc_metadata = meta_obj
                else:
                    # Defensive conversion of custom metadata object to dict
                    doc_metadata = {k: v for k, v in vars(meta_obj).items() if v is not None}
            except Exception as e:
                logger.warning("Failed to extract structural metadata from doc.metadata: %s", e)

        extract_start = time.perf_counter()
        for element, _ in doc.iterate_items():
            etype = type(element).__name__
            pno   = _page_no(element)

            # ── Title (structural classification by Docling) ───────────────────
            if etype == "TitleItem":
                text = (element.text or "").strip()
                if text and doc_title_guess is None:
                    doc_title_guess = text
                # Title text feeds into first_page_text for heuristic fallback
                if pno <= 1:
                    first_page_lines.append(text)

            # ── Section header ─────────────────────────────────────────────────
            elif etype == "SectionHeaderItem":
                text = (element.text or "").strip()
                if text:
                    current_section = text
                if pno <= 1:
                    first_page_lines.append(text)

            # ── Text block ────────────────────────────────────────────────────
            elif etype == "TextItem":
                text = (element.text or "").strip()
                if not text or len(text) < _MIN_TEXT_LEN:
                    continue

                if pno <= 1:
                    first_page_lines.append(text)

                blocks.append(ParsedBlock(
                    content_type  = "text",
                    content       = text,
                    page_number   = pno,
                    section_title = current_section,
                ))

            # ── Equation ──────────────────────────────────────────────────────
            elif etype == "EquationItem":
                # Prefer LaTeX export; fall back to plain text
                text = ""
                try:
                    text = element.text or ""
                    if hasattr(element, "export_to_latex"):
                        latex = element.export_to_latex(doc) or ""
                        if latex.strip():
                            text = latex.strip()
                except Exception:
                    pass
                text = text.strip()
                if not text:
                    continue

                blocks.append(ParsedBlock(
                    content_type  = "equation",
                    content       = text,
                    page_number   = pno,
                    section_title = current_section,
                ))

            # ── Table ─────────────────────────────────────────────────────────
            elif etype == "TableItem":
                md = ""
                try:
                    md = element.export_to_markdown(doc) or ""
                    md = md.strip()
                except Exception as e:
                    logger.warning("Table markdown export failed on page %d: %s", pno, e)
                    continue

                if not md:
                    continue

                # Count data rows (lines that start with "|" and are not separator rows)
                sep_re   = re.compile(r"^\|[\s\-\|:]+\|?\s*$")
                data_rows = [
                    ln for ln in md.splitlines()
                    if ln.strip().startswith("|") and not sep_re.match(ln.strip())
                ]
                if len(data_rows) < _MIN_TABLE_ROWS:
                    continue

                table_counter += 1
                label          = f"Table {table_counter}"

                # Build a rich text summary: label + header row + all data rows
                embeddable_content = _table_to_embeddable(md, label)

                blocks.append(ParsedBlock(
                    content_type   = "table",
                    content        = embeddable_content,
                    page_number    = pno,
                    section_title  = current_section,
                    figure_number  = label,
                    table_markdown = md,
                ))

            # ── Figure / Picture ──────────────────────────────────────────────
            elif etype == "PictureItem":
                figure_start = time.perf_counter()
                

                figure_counter += 1
                label           = f"Figure {figure_counter}"

                saved_path = _save_figure(element, doc, pdf_path, figure_counter, figures_path)
                caption    = _extract_caption(element, figure_counter, pno)

                blocks.append(ParsedBlock(
                    content_type  = "figure",
                    content       = caption,
                    page_number   = pno,
                    section_title = current_section,
                    figure_number = label,
                    image_path    = saved_path,
                ))

                logger.info(
                    "%s processed in %.2fs",
                    label,
                    time.perf_counter() - figure_start,
                )

            # ── Standalone caption ────────────────────────────────────────────
            elif etype == "CaptionItem":
                text = (element.text or "").strip()
                if len(text) < _MIN_PARA_LEN:
                    continue
                # Attach as a caption block so chunker can merge with parent if needed
                blocks.append(ParsedBlock(
                    content_type  = "caption",
                    content       = text,
                    page_number   = pno,
                    section_title = current_section,
                ))


        logger.info(
            "Element extraction took %.2fs",
            time.perf_counter() - extract_start,
        )
    finally:
        if doc is not None:
            del doc
        if result is not None:
            del result
        gc.collect()

    # ── Build section_map ─────────────────────────────────────────────────────
    section_map: dict[str, list[int]] = {}
    for idx, block in enumerate(blocks):
        sec = block.section_title
        section_map.setdefault(sec, []).append(idx)

    # ── first_page_text ───────────────────────────────────────────────────────
    # Join the lines we collected from page 1 elements for the metadata heuristic.
    first_page_text = "\n".join(first_page_lines)
    logger.info(
        "Total parser time %.2fs",
        time.perf_counter() - parse_start,
    )
    return ParsedDocument(
        blocks          = blocks,
        first_page_text = first_page_text,
        doc_title_guess = doc_title_guess,
        metadata       = doc_metadata,
        section_map     = section_map,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_doc() -> ParsedDocument:
    return ParsedDocument(
        blocks=[],
        first_page_text="",
        doc_title_guess=None,
        metadata={},
        section_map={}
    )


def _page_no(element) -> int:
    """Extract 1-indexed page number from a Docling element; 0 on failure."""
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
    Save a PictureItem image to disk as PNG.
    Returns absolute path string or None.
    PIL Image is always closed to free pixel buffer memory.
    """
    fig_id       = hashlib.md5(f"{pdf_path}_{figure_counter}".encode()).hexdigest()[:8]
    img_filename = f"{fig_id}.png"
    img_path     = figures_path / img_filename

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
        if img is not None:
            try:
                img.close()
            except Exception:
                pass


def _extract_caption(element, figure_counter: int, page_no: int) -> str:
    """
    Extract caption from a PictureItem using all available Docling attributes.
    Never returns empty string — placeholder ensures chunk is always embeddable.
    """
    # 1. Try 'captions' list (Modern Docling)
    if hasattr(element, "captions") and element.captions:
        for cap in element.captions:
            text = getattr(cap, "text", None) or str(cap)
            if text and text.strip():
                return text.strip()

    # 2. Try direct 'caption' attribute (Legacy/Fallback)
    if hasattr(element, "caption") and element.caption:
        text = str(element.caption).strip()
        if text:
            return text

    # 3. Fallback placeholder
    return f"Figure {figure_counter} on page {page_no}"


def _table_to_embeddable(md: str, label: str) -> str:
    """
    Build a plain-English-friendly string from a markdown table that embeds well.

    Format:
        Table N — <header columns joined by ' | '>
        <all data rows as pipe-separated text>

    This gives the embedding model both the column semantics and the cell values
    without raw markdown pipe syntax dominating the representation.
    """
    lines = md.splitlines()
    if not lines:
        return label

    # Identify header row and separator row
    sep_re     = re.compile(r"^\|[\s\-\|:]+\|?\s*$")
    header_row = None
    data_lines = []

    for _, ln in enumerate(lines):
        stripped = ln.strip()
        if not stripped:
            continue
        if sep_re.match(stripped):
            continue
        if header_row is None:
            header_row = stripped
        else:
            data_lines.append(stripped)

    def _parse_cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip("|").split("|") if c.strip()]

    parts = [label]
    if header_row:
        cols = _parse_cells(header_row)
        parts.append("Columns: " + " | ".join(cols))

    for dl in data_lines:
        cells = _parse_cells(dl)
        if cells:
            parts.append(" | ".join(cells))

    return "\n".join(parts)
