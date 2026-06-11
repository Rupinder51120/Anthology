"""
Document parser using Docling.
Extracts: text blocks, tables, figures, captions, equations.
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedBlock:
    content_type: str          # text | table | figure | equation | caption
    content: str               # primary text content
    page_number: int
    section_title: str
    figure_number: str | None = None
    image_path: str | None = None
    table_markdown: str | None = None
    metadata: dict = field(default_factory=dict)


def parse_pdf(pdf_path: str, figures_dir: str = "data/figures") -> list[ParsedBlock]:
    """
    Parse PDF using Docling. Returns list of ParsedBlock objects.
    Falls back to PyMuPDF if Docling unavailable.
    """
    try:
        return _parse_with_docling(pdf_path, figures_dir)
    except ImportError:
        print("Docling not available, falling back to PyMuPDF")
        return _parse_with_pymupdf(pdf_path)
    except Exception as e:
        print(f"Docling parse failed: {e}, falling back to PyMuPDF")
        return _parse_with_pymupdf(pdf_path)


def _parse_with_docling(pdf_path: str, figures_dir: str) -> list[ParsedBlock]:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path)
    doc = result.document
    blocks: list[ParsedBlock] = []
    current_section = "preamble"
    figure_counter = 0
    table_counter = 0

    for element, _ in doc.iterate_items():
        element_type = type(element).__name__

        if element_type == "SectionHeaderItem":
            current_section = element.text.strip()

        elif element_type == "TextItem":
            text = element.text.strip()
            if not text or len(text) < 20:
                continue
            blocks.append(ParsedBlock(
                content_type="text",
                content=text,
                page_number=element.prov[0].page_no if element.prov else 0,
                section_title=current_section,
            ))

        elif element_type == "TableItem":
            table_counter += 1
            md = element.export_to_markdown(doc)
            blocks.append(ParsedBlock(
                content_type="table",
                content=md,
                page_number=element.prov[0].page_no if element.prov else 0,
                section_title=current_section,
                figure_number=f"Table {table_counter}",
                table_markdown=md,
            ))

        elif element_type == "PictureItem":
            figure_counter += 1
            fig_id = hashlib.md5(f"{pdf_path}_{figure_counter}".encode()).hexdigest()[:8]
            img_filename = f"{fig_id}.png"
            img_path = figures_path / img_filename

            try:
                img = element.get_image(doc)
                if img:
                    img.save(str(img_path))
            except Exception:
                img_path = None

            caption = ""
            if hasattr(element, "caption") and element.caption:
                caption = element.caption.strip()

            blocks.append(ParsedBlock(
                content_type="figure",
                content=caption or f"Figure {figure_counter} from page {element.prov[0].page_no if element.prov else 0}",
                page_number=element.prov[0].page_no if element.prov else 0,
                section_title=current_section,
                figure_number=f"Figure {figure_counter}",
                image_path=str(img_path) if img_path and img_path.exists() else None,
            ))

    return blocks


def _parse_with_pymupdf(pdf_path: str) -> list[ParsedBlock]:
    import fitz
    doc = fitz.open(pdf_path)
    blocks = []
    current_section = "preamble"

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        for para in paragraphs:
            if re.match(r"^\d+\.\s+[A-Z]|^[A-Z][A-Z\s]{3,}$", para):
                current_section = para[:100]
                continue
            blocks.append(ParsedBlock(
                content_type="text",
                content=para,
                page_number=page_num,
                section_title=current_section,
            ))

    doc.close()
    return blocks
