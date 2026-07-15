"""
src/pipeline_runner.py

GATE FIX vs prior version:
The old runner called:
    chunks = _retriever_module.retrieve(question, top_k=top_k, use_hyde=use_hyde)
    result = generate_answer(question, chunks)

Both calls are unawaited sync calls into functions that are now async
(`retriever.retrieve` and `generator.generate_answer`), and `retrieve()`
now requires a `db` session (raises ValueError if None) plus accepts a
`strategy` param. This file predates that refactor and would raise
immediately if run. Rewritten below:
  - fully async, run via asyncio.run() at the entrypoint
  - opens an AsyncSessionLocal() per question, matching benchmark.py's
    _run_one() pattern (one session per query, not one shared session
    held open across the whole run)
  - accepts `strategy`, defaulting to "hybrid_rerank" (current production
    behavior), so this runner and benchmark.py's comparison job exercise
    the same retrieval code path instead of two divergent ones
"""

import asyncio
import json
import time
import argparse
from pathlib import Path

import src.retrieval.retriever as _retriever_module
from src.generation.generator import generate_answer
from api.core.database import AsyncSessionLocal
from src.evaluation.evaluator import evaluate_results, save_scores


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


# ─── per-question execution ───────────────────────────────────

async def _run_one(question: str, top_k: int, use_hyde: bool, strategy: str) -> dict:
    t_start = time.time()
    async with AsyncSessionLocal() as db:
        chunks = await _retriever_module.retrieve(
            question, top_k=top_k, use_hyde=use_hyde, db=db, strategy=strategy
        )
    result = await generate_answer(question, chunks)
    return {
        "answer":    result["answer"],
        "contexts":  [c["text"] for c in chunks],
        "sources":   [c["metadata"]["source"] for c in chunks],
        # chunk-level ground truth support, same as benchmark.py's _run_one
        "chunk_ids": [c["metadata"]["chunk_id"] for c in chunks],
        "elapsed_s": round(time.time() - t_start, 2),
    }


# ─── main runner (async) ───────────────────────────────────────

async def run_pipeline_on_dataset_async(
    qa_path: str = "indexes/qa_dataset.json",
    output_path: str = "indexes/pipeline_results.json",
    use_hyde: bool = False,
    top_k: int = 5,
    strategy: str = "hybrid_rerank",
    sample_size: int | None = None,
    run_judge: bool = False,
    sleep_between: float = 0.0,
) -> list[dict]:

    checkpoint_path = output_path.replace(".json", "_checkpoint.json")

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    if sample_size is not None:
        qa_pairs = qa_pairs[:sample_size]

    # Fix (checkpoint): a fully-completed checkpoint from a previous run
    # makes `remaining` empty -> 0 questions processed -> empty output file.
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
    print(f"Config: HyDE={use_hyde}, top_k={top_k}, strategy={strategy}\n")

    for i, qa in enumerate(remaining):
        question         = qa["question"]
        ground_truth     = qa["answer"]
        source_chunk     = qa.get("source_chunk", "")
        source_paper     = qa.get("source_paper", source_chunk)
        source_chunk_id  = qa.get("source_chunk_id", "")
        idx              = len(results) + 1
        total            = len(qa_pairs)

        print(f"[{idx}/{total}] {question[:70]}...")
        t_start = time.time()

        try:
            r = await _run_one(question, top_k=top_k, use_hyde=use_hyde, strategy=strategy)
            results.append({
                "question":        question,
                "ground_truth":    ground_truth,
                "source_chunk":    source_chunk,
                "source_paper":    source_paper,
                "source_chunk_id": source_chunk_id,
                **r,
                "config": {"hyde": use_hyde, "top_k": top_k, "strategy": strategy},
            })

        except Exception as e:
            print(f"  Pipeline failed: {e}")
            results.append({
                "question":        question,
                "ground_truth":    ground_truth,
                "source_chunk":    source_chunk,
                "source_paper":    source_paper,
                "source_chunk_id": source_chunk_id,
                "answer":          "ERROR",
                "contexts":        [],
                "sources":         [],
                "chunk_ids":       [],
                "config":          {"hyde": use_hyde, "top_k": top_k, "strategy": strategy},
                "elapsed_s":       round(time.time() - t_start, 2),
                "error":           str(e),
            })

        _save_checkpoint(results, checkpoint_path)

        if i < len(remaining) - 1 and sleep_between > 0:
            await asyncio.sleep(sleep_between)

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- End-to-End Evaluation Link ---
    label = f"{strategy}_top{top_k}"
    metrics = evaluate_results(results, label=label, run_judge=run_judge)

    metrics_path = output_path.replace(".json", f"_{strategy}_{top_k}_scores.json")
    save_scores(metrics, path=metrics_path)
    # ----------------------------------

    cp = Path(checkpoint_path)
    if cp.exists():
        cp.unlink()

    print(f"\nDone. {len(results)} results saved → {output_path}")
    return results


def run_pipeline_on_dataset(
    qa_path: str = "indexes/qa_dataset.json",
    output_path: str = "indexes/pipeline_results.json",
    use_hyde: bool = False,
    top_k: int = 5,
    strategy: str = "hybrid_rerank",
    sample_size: int | None = None,
    run_judge: bool = False,
    sleep_between: float = 0.0,
) -> list[dict]:
    """
    Sync entrypoint for CLI/script use (`python -m src.pipeline_runner`).
    Do NOT call this from inside an already-running event loop (e.g. from
    a FastAPI route) — use run_pipeline_on_dataset_async() and await it
    there instead, same reasoning as benchmark.py's async-job fix.
    """
    return asyncio.run(
        run_pipeline_on_dataset_async(
            qa_path=qa_path,
            output_path=output_path,
            use_hyde=use_hyde,
            top_k=top_k,
            strategy=strategy,
            sample_size=sample_size,
            run_judge=run_judge,
            sleep_between=sleep_between,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-path", type=str, default="indexes/qa_dataset.json")
    parser.add_argument("--output-path", type=str, default="indexes/pipeline_results.json")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--strategy", type=str, default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    run_pipeline_on_dataset(
        qa_path=args.qa_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        strategy=args.strategy,
        top_k=args.top_k,
    )
