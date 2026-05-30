import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from rank_bm25 import BM25Okapi


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Use IVF index for larger datasets (>10k chunks),
    flat index for smaller ones. Auto-selects.
    """
    n, dim = embeddings.shape
    embeddings = embeddings.astype("float32")

    if n < 1000:
        # flat exact search — perfect for small datasets
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"FAISS FlatIP index built: {index.ntotal} vectors, dim={dim}")
    else:
        # IVF approximate search — faster for large datasets
        nlist  = min(int(np.sqrt(n)), 256)
        quant  = faiss.IndexFlatIP(dim)
        index  = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(nlist // 4, 32)
        print(f"FAISS IVFFlat index built: {index.ntotal} vectors, "
              f"dim={dim}, nlist={nlist}, nprobe={index.nprobe}")

    return index


def save_faiss_index(index: faiss.Index,
                     path: str = "indexes/faiss_index.bin"):
    Path("indexes").mkdir(exist_ok=True)
    faiss.write_index(index, path)
    print(f"FAISS index saved → {path}")


def load_faiss_index(path: str = "indexes/faiss_index.bin") -> faiss.Index:
    if not Path(path).exists():
        raise FileNotFoundError(f"FAISS index not found at {path}. Run build_index.py first.")
    return faiss.read_index(path)


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Build BM25 with improved tokenization:
    - lowercase
    - remove punctuation
    - remove single chars
    """
    import re
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
    """Return stats about current indexes for dashboard."""
    stats = {}

    faiss_path = Path("indexes/faiss_index.bin")
    if faiss_path.exists():
        index = load_faiss_index()
        stats["faiss_vectors"] = index.ntotal
        stats["faiss_size_mb"] = round(faiss_path.stat().st_size / 1e6, 2)

    chunks_path = Path("indexes/chunks_metadata.json")
    if chunks_path.exists():
        with open(chunks_path) as f:
            chunks = json.load(f)
        stats["total_chunks"] = len(chunks)
        stats["unique_papers"] = len(set(c["metadata"]["source"] for c in chunks))

    return stats