import json
import re
from datetime import datetime
from pathlib import Path

INDEXES_DIR     = "indexes"
CHECKPOINT_PATH = "indexes/checkpoint.json"


def preserve_math(text: str) -> str:
    text = re.sub(r'\$\$([^$]+)\$\$', r'[MATH_BLOCK]\1[/MATH_BLOCK]', text)
    text = re.sub(r'\$([^$]+)\$',     r'[MATH]\1[/MATH]', text)
    text = re.sub(r'(\\\\[a-zA-Z]+\\{[^}]*\\})', r'[MATH]\1[/MATH]', text)
    return text


def load_checkpoint() -> dict:
    if Path(CHECKPOINT_PATH).exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        if "processed_files" not in data:
            data["processed_files"] = []
        return data
    return {"processed_files": [], "last_run": None}


def save_checkpoint(checkpoint: dict):
    Path(INDEXES_DIR).mkdir(exist_ok=True)
    checkpoint["last_run"] = datetime.now().isoformat()
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def filter_chunks(chunks: list, score_fn, min_score: float = 0.3) -> tuple:
    scored  = [(chunk, score_fn(chunk)) for chunk in chunks]
    kept    = [c for c, s in scored if s >= min_score]
    removed = [c for c, s in scored if s < min_score]
    for chunk, score in scored:
        chunk["metadata"]["quality_score"] = round(score, 3)
    stats = {
        "total":        len(chunks),
        "kept":         len(kept),
        "removed":      len(removed),
        "removal_rate": f"{len(removed)/len(chunks)*100:.1f}%" if chunks else "0%",
    }
    return kept, stats
