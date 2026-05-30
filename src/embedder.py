import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# SPECTER2 is trained on academic citation graphs
# best model for research paper similarity
MODEL_NAME = "allenai-specter"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"Model loaded. Embedding dim: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        raise ValueError("No texts to embed")

    model = get_model()

    # clean texts before embedding
    cleaned = []
    for t in texts:
        t = t.strip()
        if not t:
            t = "[EMPTY]"
        cleaned.append(t[:2048])  # SPECTER max input

    print(f"Embedding {len(cleaned)} texts (batch_size={batch_size})...")

    embeddings = model.encode(
        cleaned,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize → cosine sim = dot product
    )

    return embeddings.astype("float32")


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> np.ndarray:
    # prioritize high-value sections slightly by prepending title
    texts = []
    for c in chunks:
        prefix = f"{c['metadata']['title']}. {c['metadata']['section']}. "
        text   = prefix + c["text"]
        texts.append(text)

    return embed_texts(texts, batch_size=batch_size)


def embed_papers_for_recommendation(
    all_papers: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    """
    Embed each paper at the document level using abstract + title.
    Used for paper-to-paper similarity in recommendation engine.
    """
    paper_texts = []
    paper_meta  = []

    for paper in all_papers:
        title    = paper["metadata"]["title"]
        abstract = paper["sections"].get("abstract", "")

        if not abstract:
            abstract = paper["full_text"][:600]

        # SPECTER was trained on title + abstract pairs
        text = f"{title} [SEP] {abstract[:400]}"
        paper_texts.append(text)
        paper_meta.append(paper["metadata"])

    embeddings = embed_texts(paper_texts, batch_size=16)
    return embeddings, paper_meta


def save_embeddings(embeddings: np.ndarray, path: str):
    Path("indexes").mkdir(exist_ok=True)
    np.save(path, embeddings)
    print(f"Saved embeddings → {path} shape={embeddings.shape} dtype={embeddings.dtype}")


def load_embeddings(path: str) -> np.ndarray:
    arr = np.load(path)
    return arr.astype("float32")


if __name__ == "__main__":
    from src.ingest import load_all_papers
    from src.chunker import chunk_all_papers

    papers = load_all_papers("data/papers")
    chunks = chunk_all_papers(papers)

    chunk_embs = embed_chunks(chunks)
    save_embeddings(chunk_embs, "indexes/chunk_embeddings.npy")

    paper_embs, meta = embed_papers_for_recommendation(papers)
    save_embeddings(paper_embs, "indexes/paper_embeddings.npy")

    print(f"\nDone. Chunk embs: {chunk_embs.shape} | Paper embs: {paper_embs.shape}")