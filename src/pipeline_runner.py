"""
src/pipeline_runner.py
"""

import json
import time
from pathlib import Path

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from groq import RateLimitError

import src.retriever as _retriever_module
from src.hyde import expand_query_with_hyde
from src.generator import generate_answer


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


# ─── retry-wrapped retrieve + generate ───────────────────────

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _retrieve_with_retry(search_query: str, top_k: int) -> list[dict]:
    return _retriever_module.retrieve(search_query, top_k=top_k)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _generate_with_retry(question: str, chunks: list[dict]) -> dict:
    return generate_answer(question, chunks)


# ─── main runner ─────────────────────────────────────────────

def run_pipeline_on_dataset(
    qa_path: str = "indexes/qa_dataset.json",
    output_path: str = "indexes/pipeline_results.json",
    use_hyde: bool = True,
    top_k: int = 5,
    sleep_between: float = 1.0,
) -> list[dict]:

    checkpoint_path = output_path.replace(".json", "_checkpoint.json")

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    done_map = _load_checkpoint(checkpoint_path)
    results  = list(done_map.values())
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
            search_query = expand_query_with_hyde(question) if use_hyde else question
            chunks       = _retrieve_with_retry(search_query, top_k=top_k)
            result       = _generate_with_retry(question, chunks)

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

        except RateLimitError as e:
            print(f"\n  Rate limit exhausted after retries: {e}")
            print(f"  Checkpointing {len(results)} results and stopping.")
            _save_checkpoint(results, checkpoint_path)
            raise

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

        if i < len(remaining) - 1:
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
