import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

import os
MODEL_NAME = "allenai/specter2_base"

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
       # FIX (audit #2): model is SPECTER2 (allenai/specter2_base), not
        # MiniLM as the old comment claimed. SPECTER2 supports 512 tokens
        # (~2000-2500 chars). The old 1024-char limit truncated 13.64% of
        # chunks by an average of 25.2% (measured on the 12,679-chunk
        # corpus), discarding real content the model could have used.
        cleaned.append(t[:2000])

    embeddings = model.encode(
        cleaned,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> np.ndarray:
    content_type_prefix = {
        "text":     "",
        "table":    "Table: ",
        "figure":   "Figure: ",
        "equation": "Equation: ",
    }
    texts = []
    for c in chunks:
        meta   = c["metadata"]
        ctype  = meta.get("content_type", "text")
        prefix = content_type_prefix.get(ctype, "")
        if ctype == "text":
            prefix = f"{meta['title']}. {meta.get('section', '')}. "
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