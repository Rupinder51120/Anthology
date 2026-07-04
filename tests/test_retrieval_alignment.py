from types import SimpleNamespace

from src.retrieval import embedder
from src.retrieval.retriever import _row_to_dict


class DummyRow(SimpleNamespace):
    pass


def test_row_to_dict_preserves_new_chunk_metadata():
    row = DummyRow(
        text="Example text",
        chunk_id="abc123",
        source="paper.pdf",
        title="Paper",
        authors="Author One",
        year=2024,
        section="Methods",
        section_priority=0.95,
        chunk_type="narrative",
        content_type="text",
        page_number=3,
        figure_number=None,
        image_path=None,
        table_markdown=None,
        table_summary=None,
        similarity=0.87,
        chunk_index=4,
        char_count=123,
        word_count=20,
        is_enriched=True,
    )

    result = _row_to_dict(row)

    metadata = result["metadata"]
    assert metadata["chunk_index"] == 4
    assert metadata["char_count"] == 123
    assert metadata["word_count"] == 20
    assert metadata["is_enriched"] is True


def _capture_embedding_text(monkeypatch, chunks):
    captured = {}

    def fake_embed_texts(texts, batch_size=32):
        captured["texts"] = texts
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(embedder, "embed_texts", fake_embed_texts)
    embedder.embed_chunks(chunks)
    return captured["texts"][0]


def test_embed_chunks_uses_enriched_table_context(monkeypatch):
    chunks = [
        {
            "text": "table content",
            "metadata": {
                "title": "Paper",
                "section": "Results",
                "content_type": "table",
                "table_summary": "A concise table summary",
            },
        }
    ]

    text = _capture_embedding_text(monkeypatch, chunks)

    assert "Paper" in text
    assert "A concise table summary" in text


def test_table_embedding_contract_full(monkeypatch):
    """
    Explicit contract check for table chunks: title, section, table_summary,
    table_markdown, and the semantic table text must all be present, while
    operational/ranking-only metadata (is_enriched, etc.) must never be
    embedded.
    """
    chunks = [
        {
            "text": "[Results] | col1 | col2 |\n|---|---|\n| 1 | 2 |",
            "metadata": {
                "title": "Paper",
                "section": "Results",
                "content_type": "table",
                "table_summary": "A concise table summary",
                "table_markdown": "| col1 | col2 |\n|---|---|\n| 1 | 2 |",
                "is_enriched": True,
                "section_priority": 0.95,
                "chunk_index": 2,
                "char_count": 40,
                "word_count": 8,
                "chunk_id": "deadbeef",
                "source": "paper.pdf",
            },
        }
    ]

    text = _capture_embedding_text(monkeypatch, chunks)

    assert "Paper" in text
    assert "Results" in text
    assert "A concise table summary" in text
    assert "| col1 | col2 |" in text
    assert "| 1 | 2 |" in text

    # Operational / ranking fields must never leak into embedded text
    for forbidden in ("is_enriched", "True", "0.95", "deadbeef", "paper.pdf"):
        assert forbidden not in text


def test_table_embedding_fallback_no_summary(monkeypatch):
    chunks = [
        {
            "text": "| col1 | col2 |\n|---|---|\n| 1 | 2 |",
            "metadata": {
                "title": "Paper",
                "section": "Results",
                "content_type": "table",
                "table_markdown": "| col1 | col2 |\n|---|---|\n| 1 | 2 |",
            },
        }
    ]

    text = _capture_embedding_text(monkeypatch, chunks)

    assert "Paper" in text
    assert "Results" in text
    assert "| col1 | col2 |" in text


def test_table_embedding_fallback_no_markdown(monkeypatch):
    chunks = [
        {
            "text": "table content only",
            "metadata": {
                "title": "Paper",
                "section": "Results",
                "content_type": "table",
                "table_summary": "A concise table summary",
            },
        }
    ]

    text = _capture_embedding_text(monkeypatch, chunks)

    assert "Paper" in text
    assert "Results" in text
    assert "A concise table summary" in text
    assert "table content only" in text


def test_figure_chunk_includes_figure_number_and_caption(monkeypatch):
    chunks = [
        {
            "text": "[Results] Figure showing the model architecture diagram.",
            "metadata": {
                "title": "Paper",
                "section": "Results",
                "content_type": "figure",
                "figure_number": "Figure 3",
                "image_path": "figs/fig3.png",
            },
        }
    ]

    text = _capture_embedding_text(monkeypatch, chunks)

    assert "Figure 3" in text
    assert "Figure showing the model architecture diagram." in text
    assert "Visual figure" not in text
    # must not duplicate the word "Figure" (e.g. "Figure: Figure 3")
    assert "Figure: Figure 3" not in text


def test_figure_number_does_not_leak_into_text_chunks(monkeypatch):
    chunks = [
        {
            "text": "[Introduction] This is the semantic body of the chunk.",
            "metadata": {
                "title": "Paper",
                "section": "Introduction",
                "content_type": "text",
                "figure_number": "Figure 3",
            },
        }
    ]

    text = _capture_embedding_text(monkeypatch, chunks)

    assert "Figure" not in text
    assert "This is the semantic body of the chunk." in text


def test_text_chunk_embedding_still_valid(monkeypatch):
    chunks = [
        {
            "text": "[Introduction] This is the semantic body of the chunk.",
            "metadata": {
                "title": "Paper",
                "section": "Introduction",
                "content_type": "text",
                "page_number": 3,
            },
        }
    ]

    text = _capture_embedding_text(monkeypatch, chunks)

    assert "Paper" in text
    assert "Introduction" in text
    assert "This is the semantic body of the chunk." in text
    # section prefix baked into chunk["text"] should not be duplicated
    assert text.count("Introduction") == 1
    # page_number is metadata-only, never embedded
    assert "Page:" not in text
    assert "3" not in text