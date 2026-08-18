"""
Regression test for the metadata_resolver.py `re.search` crash (audit BUG-01):
resolve_metadata() used re.search() on Docling's `date` field with no `import re`
in the module, causing a guaranteed NameError on any PDF where Docling populates
a truthy `date` metadata value.
"""
from src.ingestion.metadata_resolver import resolve_metadata
from src.ingestion.parser import ParsedDocument


def _doc_with_date(date_str: str) -> ParsedDocument:
    return ParsedDocument(
        blocks=[],
        first_page_text="Some Paper Title\nJane Doe, John Smith\nAbstract text.",
        doc_title_guess=None,
        metadata={
            "title": "Some Paper Title",
            "authors": ["Jane Doe", "John Smith"],
            "date": date_str,
        },
    )


def test_resolve_metadata_extracts_year_from_date_field():
    doc = _doc_with_date("2026-03-14")
    result = resolve_metadata(doc, filename="__regression_test_paper.pdf")
    assert result["metadata_source"] == "docling_structural"
    assert result["year"] == "2026"


def test_resolve_metadata_handles_date_with_no_year_match():
    # No 19xx/20xx pattern present in the date string — re.search() must not raise,
    # and the resolved year must fall back to the heuristic extractor's default
    # rather than being overwritten by a bogus match.
    doc = _doc_with_date("unknown")
    result = resolve_metadata(doc, filename="__regression_test_paper_2.pdf")
    assert result["year"] == "Unknown"


def test_resolve_metadata_handles_missing_date_field():
    # No "date" key at all in Docling metadata — the re.search() branch must be
    # skipped entirely without raising.
    doc = ParsedDocument(
        blocks=[],
        first_page_text="Some Paper Title\nJane Doe",
        doc_title_guess=None,
        metadata={"title": "Some Paper Title", "authors": ["Jane Doe"]},
    )
    result = resolve_metadata(doc, filename="__regression_test_paper_3.pdf")
    assert result["year"] == "Unknown"
