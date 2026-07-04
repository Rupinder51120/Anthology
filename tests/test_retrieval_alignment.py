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


def test_embed_chunks_uses_enriched_table_context(monkeypatch):
    captured = {}

    def fake_embed_texts(texts, batch_size=32):
        captured["texts"] = texts
        return [[0.1, 0.2]]

    monkeypatch.setattr(embedder, "embed_texts", fake_embed_texts)

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

    embedder.embed_chunks(chunks)

    assert "Paper" in captured["texts"][0]
    assert "A concise table summary" in captured["texts"][0]
