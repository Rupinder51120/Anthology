"""
SPECTER2 embedder for all modalities.
Singleton model load — loaded once per worker process.
"""
from __future__ import annotations
import numpy as np

MODEL_NAME = "allenai/specter2_base"
_MODEL = None
_TOKENIZER = None


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from transformers import AutoTokenizer, AutoModel
        import torch
        print(f"Loading {MODEL_NAME}...")
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModel.from_pretrained(MODEL_NAME)
        _MODEL.eval()
        print("SPECTER2 loaded.")
    return _MODEL, _TOKENIZER


def embed_texts(texts: list[str], batch_size: int = 8) -> np.ndarray:
    import torch
    model, tokenizer = _load_model()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings).astype("float32")


def embed_chunk(chunk: dict) -> np.ndarray:
    """Embed a single chunk. Prepends content_type context."""
    prefix_map = {
        "text":     "",
        "table":    "Table: ",
        "figure":   "Figure: ",
        "equation": "Equation: ",
        "caption":  "Caption: ",
    }
    prefix = prefix_map.get(chunk["content_type"], "")
    text = prefix + chunk["content"]
    return embed_texts([text])[0]
