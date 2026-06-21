import json
import re
import os
from datetime import datetime
from pathlib import Path

INDEXES_DIR     = "indexes"
CHECKPOINT_PATH = "indexes/checkpoint.json"


def preserve_math(text: str) -> str:
    text = re.sub(r'\$\$([^$]+)\$\$', r'[MATH_BLOCK]\1[/MATH_BLOCK]', text)
    text = re.sub(r'\$([^$]+)\$',     r'[MATH]\1[/MATH]', text)
    text = re.sub(r'(\\[a-zA-Z]+(?:\{[^}]*\}\s*){1,2})', r'[MATH]\1[/MATH]', text)
    return text


def load_checkpoint() -> dict:
    path = Path(CHECKPOINT_PATH)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Checkpoint data is not a dictionary")
            if "processed_files" not in data:
                data["processed_files"] = []
            return data
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"Warning: Checkpoint corrupted or invalid ({e}). Resetting.")
    return {"processed_files": [], "last_run": None}


def save_checkpoint(checkpoint: dict):
    Path(INDEXES_DIR).mkdir(exist_ok=True)

    # Avoid mutating caller-owned object
    data = checkpoint.copy()
    data["last_run"] = datetime.now().isoformat()

    # Atomic write: write to temp file then rename
    temp_path = f"{CHECKPOINT_PATH}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, CHECKPOINT_PATH)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        if Path(temp_path).exists():
            Path(temp_path).unlink()


def filter_chunks(chunks: list, score_fn, min_score: float = 0.3) -> tuple:
    kept = []
    removed = []

    for chunk in chunks:
        score = score_fn(chunk)
        # Metadata safety: ensure metadata dict exists
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        chunk["metadata"]["quality_score"] = round(score, 3)

        if score >= min_score:
            kept.append(chunk)
        else:
            removed.append(chunk)

    stats = {
        "total":        len(chunks),
        "kept":         len(kept),
        "removed":      len(removed),
        "removal_rate": f"{len(removed)/len(chunks)*100:.1f}%" if chunks else "0%",
    }
    return kept, stats
