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


def _strip_section_prefix(chunk_text: str, section: str) -> str:
    """
    chunker.py bakes a "[Section] " prefix directly into chunk["text"].
    Since the embedding text separately adds an explicit "Section: ..."
    line, keeping the bracket prefix would duplicate the section name in
    the embedded string. Stripped here only for the embedding
    representation — chunk["text"] itself (used elsewhere, e.g. for
    display/DB storage) is left untouched.
    """
    if not section:
        return chunk_text
    prefix = f"[{section}] "
    if chunk_text.startswith(prefix):
        return chunk_text[len(prefix):]
    return chunk_text


def _build_embedding_text(chunk: dict) -> str:
    """
    Builds the text actually fed to the embedding model.

    Deliberately excludes Title/Authors/Year: those fields are IDENTICAL for
    every chunk belonging to the same paper, so including them compresses
    the effective content-signal range of the embedding (measured: two
    chunks from the same paper with unrelated content came out 97.9%
    cosine-similar when title/authors/year were included, since the
    embedding model spends most of its representational capacity encoding
    the constant prefix rather than the actually-varying content).

    Section IS included (as chunker.py's own "[Section] " prefix, already
    present in chunk["text"] -- not re-added here) because it varies chunk
    to chunk within a paper and carries real topical signal.

    A controlled n=60 test against real benchmark questions (see
    docs/ANTHOLOGY_FULL_AUDIT.md, Phase 5 of the backend-completion pass)
    showed this content-first framing improves within-paper chunk-level
    hit@1 by ~75% relative and MRR by ~64% relative versus the
    title/authors/year-prefixed version. Title/authors/year remain fully
    available via chunk metadata/DB columns for citations, display, and
    filtering -- only the embedded representation changes.
    """
    meta = chunk.get("metadata") or {}
    ctype = meta.get("content_type", "text")
    chunk_text = chunk.get("text") or ""

    section = meta.get("section", "")

    # chunker.py always bakes "[Section] " onto chunk_text before it reaches
    # the DB, so this is a no-op for real production data. Kept explicit
    # (rather than trusting the caller blindly) so this function's contract
    # is correct in isolation for any caller/test fixture.
    semantic_text = chunk_text
    if section and not semantic_text.startswith(f"[{section}]"):
        semantic_text = f"[{section}] {semantic_text}"

    parts = []

    if ctype == "table":
        table_summary = meta.get("table_summary")
        if table_summary:
            parts.append(f"Summary: {table_summary}")
        table_markdown = meta.get("table_markdown")
        if table_markdown:
            parts.append(table_markdown)
    elif ctype == "figure":
        figure_number = meta.get("figure_number")
        if figure_number:
            # chunker.py's own fallback text (see chunk_parsed_blocks) uses
            # block.figure_number as a standalone label (e.g. "Figure 3"),
            # not a bare number — so a plain "Figure: {figure_number}"
            # would read as "Figure: Figure 3". Only add the "Figure: "
            # lead-in when the value doesn't already carry that word.
            if figure_number.strip().lower().startswith("figure"):
                parts.append(figure_number)
            else:
                parts.append(f"Figure: {figure_number}")

    # Equation chunks fall through to the generic path below: their
    # semantic content is already carried in chunk_text, so no separate
    # branch is needed.

    if semantic_text:
        parts.append(semantic_text)

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