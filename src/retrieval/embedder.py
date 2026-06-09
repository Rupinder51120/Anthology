import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

import os
MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
    if os.getenv("USE_PGVECTOR", "false").lower() == "true"
    else "BAAI/bge-large-en-v1.5"
)

def get_model() -> SentenceTransformer:
    if not hasattr(get_model, "_instance"):
        print(f"Loading embedding model: {MODEL_NAME}")
        get_model._instance = SentenceTransformer(MODEL_NAME)
        print(f"Model loaded. Dim: {get_model._instance.get_sentence_embedding_dimension()}")
    return get_model._instance


def set_model(model: SentenceTransformer):
    """Allow app.py to inject Streamlit-cached model."""
    get_model._instance = model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        raise ValueError("No texts to embed")

    model   = get_model()
    cleaned = []
    for t in texts:
        t = t.strip()
        if not t:
            t = "[EMPTY]"
        # Fix #7: HyDE documents are ~250 tokens (~1000 chars); the old 512-char
        # limit cut them roughly in half before embedding.  MiniLM supports up
        # to 512 *tokens* (~2000 chars), so 1024 chars is safe for all inputs.
        cleaned.append(t[:1024])

    embeddings = model.encode(
        cleaned,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> np.ndarray:
    texts = []
    for c in chunks:
        prefix = f"{c['metadata']['title']}. {c['metadata']['section']}. "
        texts.append(prefix + c["text"])
    return embed_texts(texts, batch_size=batch_size)


def embed_papers_for_recommendation(
    all_papers: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    paper_texts = []
    paper_meta  = []

    for paper in all_papers:
        title    = paper["metadata"]["title"]
        abstract = paper["sections"].get("abstract", "")
        if not abstract:
            abstract = paper["full_text"][:600]
        text = f"{title} [SEP] {abstract[:400]}"
        paper_texts.append(text)
        paper_meta.append(paper["metadata"])

    embeddings = embed_texts(paper_texts, batch_size=16)
    return embeddings, paper_meta


def save_embeddings(embeddings: np.ndarray, path: str):
    Path("indexes").mkdir(exist_ok=True)
    np.save(path, embeddings)
    print(f"Saved → {path} shape={embeddings.shape} dtype={embeddings.dtype}")


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path).astype("float32")


if __name__ == "__main__":
    from src.ingestion.ingest import load_all_papers
    from src.ingestion.chunker import chunk_all_papers

    papers     = load_all_papers("data/papers")
    chunks     = chunk_all_papers(papers)
    chunk_embs = embed_chunks(chunks)
    save_embeddings(chunk_embs, "indexes/chunk_embeddings.npy")

    paper_embs, meta = embed_papers_for_recommendation(papers)
    save_embeddings(paper_embs, "indexes/paper_embeddings.npy")
    print(f"Done. Chunks: {chunk_embs.shape} | Papers: {paper_embs.shape}")