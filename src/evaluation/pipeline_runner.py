"""
src/pipeline_runner.py — Ollama version (no Groq, no rate limits)
"""

import json
import time
from pathlib import Path

import src.retrieval.retriever as _retriever_module
from src.generation.generator import generate_answer


# ─── checkpoint helpers ───────────────────────────────────────

def _load_checkpoint(checkpoint_path: str) -> dict:
    p = Path(checkpoint_path)
    if p.exists():
        with open(p) as f:
            completed = json.load(f)
        print(f"Resuming from checkpoint: {len(completed)} questions already done.")
        return {r["question"]: r for r in completed}
    return {}


def _save_checkpoint(results: list[dict], checkpoint_path: str):
    Path(checkpoint_path).parent.mkdir(exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(results, f, indent=2)


# ─── main runner ─────────────────────────────────────────────

def run_pipeline_on_dataset(
    qa_path: str = "indexes/qa_dataset.json",
    output_path: str = "indexes/pipeline_results.json",
    use_hyde: bool = True,
    top_k: int = 5,
    sleep_between: float = 0.0,
) -> list[dict]:

    checkpoint_path = output_path.replace(".json", "_checkpoint.json")

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    # Fix #2 (checkpoint): each config has its own checkpoint file (namespaced
    # by output_path).  But a fully-completed checkpoint from a previous run
    # makes `remaining` empty → 0 questions processed → empty output file.
    # Detect that case and wipe the stale checkpoint so the config reruns.
    # Mid-run crashes still resume correctly (checkpoint is partial).
    done_map = _load_checkpoint(checkpoint_path)
    if done_map and len(done_map) >= len(qa_pairs):
        print("  Checkpoint covers all questions — clearing stale checkpoint for a clean run.")
        Path(checkpoint_path).unlink(missing_ok=True)
        done_map = {}

    results   = list(done_map.values())
    remaining = [qa for qa in qa_pairs if qa["question"] not in done_map]

    print(f"Total questions: {len(qa_pairs)}")
    print(f"Already done:    {len(results)}")
    print(f"Remaining:       {len(remaining)}")
    print(f"Config: HyDE={use_hyde}, top_k={top_k}\n")

    for i, qa in enumerate(remaining):
        question     = qa["question"]
        ground_truth = qa["answer"]
        source_chunk = qa.get("source_chunk", "")
        idx          = len(results) + 1
        total        = len(qa_pairs)

        print(f"[{idx}/{total}] {question[:70]}...")
        t_start = time.time()

        try:
            chunks  = _retriever_module.retrieve(question, top_k=top_k, use_hyde=use_hyde)
            result  = generate_answer(question, chunks)
            elapsed = round(time.time() - t_start, 2)

            results.append({
                "question":     question,
                "ground_truth": ground_truth,
                "source_chunk": source_chunk,
                "answer":       result["answer"],
                "contexts":     [c["text"] for c in chunks],
                "sources":      [c["metadata"]["source"] for c in chunks],
                "config":       {"hyde": use_hyde, "top_k": top_k},
                "elapsed_s":    elapsed,
            })

        except Exception as e:
            print(f"  Pipeline failed: {e}")
            results.append({
                "question":     question,
                "ground_truth": ground_truth,
                "source_chunk": source_chunk,
                "answer":       "ERROR",
                "contexts":     [],
                "sources":      [],
                "config":       {"hyde": use_hyde, "top_k": top_k},
                "elapsed_s":    round(time.time() - t_start, 2),
                "error":        str(e),
            })

        _save_checkpoint(results, checkpoint_path)

        if i < len(remaining) - 1 and sleep_between > 0:
            time.sleep(sleep_between)

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    cp = Path(checkpoint_path)
    if cp.exists():
        cp.unlink()

    print(f"\nDone. {len(results)} results saved → {output_path}")
    return results


if __name__ == "__main__":
    run_pipeline_on_dataset()