import numpy as np
import pickle
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi


# ─── Pure numpy vector index (replaces FAISS on Mac ARM64) ────

class NumpyIndex:
    """
    Pure numpy flat index — same as FAISS IndexFlatIP.
    No C++ dependencies, no segfaults, works everywhere.
    For 2600 chunks this is fast enough (< 50ms per query).
    """
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype("float32")
        self.ntotal     = len(embeddings)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query   = query.astype("float32").reshape(1, -1)
        scores  = (self.embeddings @ query.T).squeeze()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return scores[top_idx], top_idx


def build_faiss_index(embeddings: np.ndarray) -> NumpyIndex:
    index = NumpyIndex(embeddings)
    print(f"Numpy index built: {index.ntotal} vectors, dim={embeddings.shape[1]}")
    return index


def save_faiss_index(index: NumpyIndex,
                     path: str = "indexes/faiss_index.bin"):
    Path("indexes").mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)
    print(f"Index saved → {path}")


def load_faiss_index(path: str = "indexes/faiss_index.bin") -> NumpyIndex:
    if not Path(path).exists():
        raise FileNotFoundError(f"Index not found at {path}. Run build_index.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── BM25 ─────────────────────────────────────────────────────

def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized = []
    for c in chunks:
        text   = c["text"].lower()
        tokens = re.findall(r'\b[a-z][a-z0-9]{1,}\b', text)
        tokenized.append(tokens)
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index built: {len(chunks)} chunks")
    return bm25


def save_bm25_index(bm25: BM25Okapi,
                    path: str = "indexes/bm25_index.pkl"):
    Path("indexes").mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved → {path}")


def load_bm25_index(path: str = "indexes/bm25_index.pkl") -> BM25Okapi:
    if not Path(path).exists():
        raise FileNotFoundError(f"BM25 index not found at {path}. Run build_index.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_index_stats() -> dict:
    stats = {}
    index_path = Path("indexes/faiss_index.bin")
    if index_path.exists():
        index = load_faiss_index()
        stats["faiss_vectors"] = index.ntotal
        stats["faiss_size_mb"] = round(index_path.stat().st_size / 1e6, 2)

    chunks_path = Path("indexes/chunks_metadata.json")
    if chunks_path.exists():
        with open(chunks_path) as f:
            chunks = json.load(f)
        stats["total_chunks"]  = len(chunks)
        stats["unique_papers"] = len(set(
            c["metadata"]["source"] for c in chunks
        ))
    return stats