
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Mocking the ParsedBlock since we don't want to import the full parser
@dataclass
class MockParsedBlock:
    content: str
    content_type: str = "text"
    section_title: Optional[str] = None
    page_number: int = 1
    figure_number: Optional[str] = None
    image_path: Optional[str] = None
    table_markdown: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    is_enriched: bool = False

# Import the components to verify
try:
    from src.ingestion.chunker import chunk_parsed_blocks
    from src.retrieval.embedder import embed_chunks, _build_embedding_text
    print("✅ Successfully imported pipeline components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def test_e2e_flow():
    print("\n--- Starting E2E Pipeline Verification ---")

    # 1. Setup Synthetic Data
    test_metadata = {
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": "2017",
        "filename": "attention.pdf"
    }

    blocks = [
        MockParsedBlock(
            content="The Transformer is a model architecture that relies solely on attention mechanisms.",
            content_type="text",
            section_title="Introduction"
        ),
        MockParsedBlock(
            content="Table 1: Performance of Transformer on WMT 2014 English-to-German translation task.",
            content_type="table",
            section_title="Results",
            metadata={"table_summary": "Transformer outperforms previous models on WMT 2014."}
        )
    ]

    # 2. Verify Ingestion (Chunking)
    print("\nTesting Ingestion...")
    chunks = chunk_parsed_blocks(blocks, test_metadata)

    assert len(chunks) >= 2, "Should have produced at least 2 chunks"

    # Check if metadata from test_metadata was propagated
    for chunk in chunks:
        meta = chunk["metadata"]
        assert meta["title"] == "Attention Is All You Need"
        assert meta["authors"] == "Vaswani et al."
        assert meta["year"] == "2017"
    print("✅ Ingestion: Metadata propagation verified")

    # 3. Verify Embedding Construction
    print("\nTesting Embedding Text Construction...")
    for chunk in chunks:
        # The internal function _build_embedding_text is what the embedder uses
        embedding_text = _build_embedding_text(chunk)
        print(f"Chunk text for embedding: {embedding_text}")

        # VERIFY THE FIX: Authors and Year must be present in the final embedding string
        assert "Authors: Vaswani et al." in embedding_text, "Author metadata missing from embedding text!"
        assert "Year: 2017" in embedding_text, "Year metadata missing from embedding text!"
    print("✅ Embedding: Metadata included in text strings verified")

    # 4. Verify Vector Generation
    print("\nTesting Vector Generation...")
    try:
        embeddings = embed_chunks(chunks)
        print(f"Generated embeddings shape: {embeddings.shape}")
        assert embeddings.ndim == 2, "Embeddings should be a 2D array"
        assert embeddings.dtype == np.float32, "Embeddings should be float32"
        # Dimension depends on the model, but usually 384, 768, or 1024
        assert embeddings.shape[1] > 0, "Embedding dimension should be positive"
        print("✅ Embedding: Vector generation verified")
    except Exception as e:
        print(f"❌ Embedding Vector Generation failed: {e}")
        raise e

    print("\n--- ALL E2E VERIFICATIONS PASSED ---")

if __name__ == "__main__":
    test_e2e_flow()
