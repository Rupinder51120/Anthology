import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from api.core.models import EMBEDDING_MODEL
import os
MODEL_NAME = EMBEDDING_MODEL

def get_model() -> SentenceTransformer:
    if not hasattr(get_model, "_instance"):
        print(f"Loading embedding model: {MODEL_NAME}")
        get_model._instance = SentenceTransformer(MODEL_NAME)
        print(f"Model loaded. Dim: {get_model._instance.get_sentence_embedding_dimension()}")
    return get_model._instance


def set_model(model: SentenceTransformer):
    """Allow app.py to inject Streamlit-cached model."""
    get_model._instance = model


def _smart_truncate(text: str, max_len: int = 2000) -> str:
    """
    Truncates text to max_len while preserving natural boundaries:
    1. For markdown tables, prefer truncating at the last newline.
    2. Otherwise, prefer the last sentence boundary (. ! ?) if within 200 chars of max_len.
    3. Finally, prefer the last word boundary (space).
    """
    if len(text) <= max_len:
        return text

    # Issue 3: Markdown Table Detection (Check for separator row |---|)
    if "|" in text and re.search(r'\|[ :\-|]+\|', text):
        idx = text.rfind("\n", 0, max_len)
        if idx != -1:
            return text[:idx].strip()

    # Issue 2: Over-Truncation Guard
    boundaries = [". ", "! ", "? "]
    best_idx = -1
    for b in boundaries:
        idx = text.rfind(b, 0, max_len)
        if idx > best_idx:
            best_idx = idx

    # Only use sentence boundary if it's within 200 chars of max_len
    if best_idx != -1 and best_idx >= (max_len - 200):
        return text[:best_idx + 1].strip()

    # Word boundary fallback
    idx = text.rfind(" ", 0, max_len)
    if idx != -1:
        return text[:idx].strip()

    return text[:max_len]


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        raise ValueError("No texts to embed")

    model   = get_model()
    cleaned = []
    for t in texts:
        t = t.strip()
        if not t:
            t = "[EMPTY]"
       
        cleaned.append(_smart_truncate(t))

    embeddings = model.encode(
        cleaned,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def _build_embedding_text(chunk: dict) -> str:
    meta = chunk.get("metadata") or {}
    ctype = meta.get("content_type", "text")
    chunk_text = chunk.get("text") or ""

    parts = []
    title = meta.get("title", "")
    section = meta.get("section", "")
    if title:
        parts.append(f"Title: {title}")
    if section:
        parts.append(f"Section: {section}")

    page_number = meta.get("page_number")
    if page_number is not None:
        parts.append(f"Page: {page_number}")

    figure_number = meta.get("figure_number")
    if figure_number:
        parts.append(f"Figure: {figure_number}")

    if ctype == "table":
        table_summary = meta.get("table_summary")
        if table_summary:
            parts.append(f"Summary: {table_summary}")
    elif ctype == "figure" and meta.get("image_path"):
        parts.append("Visual figure")

    if chunk_text:
        parts.append(chunk_text)

    return " | ".join(part for part in parts if part)


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> np.ndarray:
    texts = [_build_embedding_text(c) for c in chunks]
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