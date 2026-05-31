"""
src/evaluator.py

Complete rewrite. RAGAS removed entirely.

Two evaluation layers:
  1. Retrieval metrics  — Hit@K, MRR, nDCG@5  (no LLM needed)
  2. Groq-as-judge      — Faithfulness, Relevance, Completeness (1-5 each)

Retrieval metrics require that each QA entry has a "source_chunk" field
(the filename of the paper the question was generated from).
pipeline_runner.py now carries this field through from the QA dataset.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from groq import Groq, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─── retrieval metrics ────────────────────────────────────────

def _hit_at_k(sources: list[str], expected_source: str, k: int) -> int:
    """1 if expected_source appears in top-k retrieved sources, else 0."""
    return int(any(expected_source in s for s in sources[:k]))


def _reciprocal_rank(sources: list[str], expected_source: str) -> float:
    for i, s in enumerate(sources):
        if expected_source in s:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(sources: list[str], expected_source: str, k: int = 5) -> float:
    relevance = [1 if expected_source in s else 0 for s in sources[:k]]
    dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(relevance))
    idcg = 1.0  # single relevant document
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(results: list[dict], k_values: list[int] = [1, 3, 5]) -> dict:
    """
    Computes Hit@K, MRR, nDCG@5 over all results that have a source_chunk field.

    Args:
        results:   Output from run_pipeline_on_dataset.
        k_values:  K values for Hit@K.

    Returns:
        Dict of metric_name -> mean value.
    """
    valid = [
        r for r in results
        if r.get("source_chunk") and r.get("sources") and r["answer"] != "ERROR"
    ]

    if not valid:
        print("  No results with source_chunk field — retrieval metrics skipped.")
        print("  Make sure qa_dataset.json has 'source_chunk' per entry.")
        return {}

    hit_scores    = {k: [] for k in k_values}
    mrr_scores    = []
    ndcg_scores   = []

    for r in valid:
        expected = r["source_chunk"]   # e.g. "Towards_Imperceptible_and_Robust..."
        sources  = r["sources"]        # list of retrieved filenames

        for k in k_values:
            hit_scores[k].append(_hit_at_k(sources, expected, k))

        mrr_scores.append(_reciprocal_rank(sources, expected))
        ndcg_scores.append(_ndcg_at_k(sources, expected, k=5))

    metrics = {f"hit@{k}": round(float(np.mean(hit_scores[k])), 4) for k in k_values}
    metrics["mrr"]    = round(float(np.mean(mrr_scores)), 4)
    metrics["ndcg@5"] = round(float(np.mean(ndcg_scores)), 4)
    metrics["n_eval"] = len(valid)

    return metrics


# ─── Groq-as-judge ───────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a RAG (Retrieval-Augmented Generation) system.

Question: {question}
Expected answer: {ground_truth}
Retrieved context (what the system had access to):
{context}
System answer: {answer}

Score each dimension from 1 to 5:
- faithfulness: Is the answer grounded in the retrieved context? (1=hallucinated, 5=fully grounded)
- relevance: Does the answer actually address the question? (1=off-topic, 5=directly answers it)
- completeness: Does the answer cover what the expected answer covers? (1=missing key points, 5=complete)

Respond ONLY with a JSON object, no explanation, no markdown:
{{"faithfulness": <int>, "relevance": <int>, "completeness": <int>}}"""


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_judge(question: str, ground_truth: str, context: str, answer: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # cheap + fast for judging
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                context=context[:1500],   # keep prompt short
                answer=answer,
            )
        }],
        temperature=0.0,
        max_tokens=60,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def compute_judge_metrics(
    results: list[dict],
    sleep_between: float = 0.5,
) -> dict:
    """
    Runs Groq-as-judge over valid results.

    Returns:
        Dict with mean faithfulness, relevance, completeness, mean_score.
    """
    valid = [
        r for r in results
        if r["answer"] not in ("ERROR", "")
        and r.get("contexts")
    ]

    if not valid:
        print("  No valid results for judge evaluation.")
        return {}

    scores = {"faithfulness": [], "relevance": [], "completeness": []}
    failed = 0

    for i, r in enumerate(valid):
        context = "\n---\n".join(r["contexts"][:3])  # top 3 chunks only
        try:
            s = _call_judge(
                question=r["question"],
                ground_truth=r["ground_truth"],
                context=context,
                answer=r["answer"],
            )
            for dim in scores:
                scores[dim].append(float(s.get(dim, 0)))
        except Exception as e:
            print(f"  Judge failed on question {i+1}: {e}")
            failed += 1

        if i < len(valid) - 1:
            time.sleep(sleep_between)

    if not scores["faithfulness"]:
        return {}

    result = {
        "faithfulness":  round(float(np.mean(scores["faithfulness"])), 4),
        "relevance":     round(float(np.mean(scores["relevance"])), 4),
        "completeness":  round(float(np.mean(scores["completeness"])), 4),
        "n_eval":        len(valid),
        "n_failed":      failed,
    }
    result["mean_score"] = round(
        float(np.mean([result["faithfulness"], result["relevance"], result["completeness"]])), 4
    )
    return result


# ─── combined eval ────────────────────────────────────────────

def evaluate_results(
    results: list[dict],
    label: str = "pipeline",
    run_judge: bool = True,
) -> dict:
    """
    Runs both retrieval metrics and (optionally) judge metrics.

    Args:
        results:   Output from run_pipeline_on_dataset.
        label:     Name for this config (used in comparison table).
        run_judge: Set False to skip Groq judge calls (saves tokens).

    Returns:
        Combined metrics dict.
    """
    print(f"\n{'='*50}")
    print(f"Evaluating: {label}")
    print(f"{'='*50}")

    output = {"label": label}

    # layer 1: retrieval metrics
    print("  Computing retrieval metrics...")
    retrieval = compute_retrieval_metrics(results)
    output["retrieval"] = retrieval
    if retrieval:
        print(f"  Hit@1={retrieval.get('hit@1'):.4f}  Hit@3={retrieval.get('hit@3'):.4f}  "
              f"Hit@5={retrieval.get('hit@5'):.4f}  MRR={retrieval.get('mrr'):.4f}  "
              f"nDCG@5={retrieval.get('ndcg@5'):.4f}  (n={retrieval.get('n_eval')})")

    # layer 2: judge metrics
    if run_judge:
        print("  Running Groq-as-judge...")
        judge = compute_judge_metrics(results)
        output["judge"] = judge
        if judge:
            print(f"  Faithfulness={judge['faithfulness']:.4f}  "
                  f"Relevance={judge['relevance']:.4f}  "
                  f"Completeness={judge['completeness']:.4f}  "
                  f"Mean={judge['mean_score']:.4f}  (n={judge['n_eval']})")
    else:
        output["judge"] = {}

    # mean_score for benchmark comparison = judge mean_score if available,
    # else MRR as proxy
    if output["judge"].get("mean_score") is not None:
        output["mean_score"] = output["judge"]["mean_score"]
    elif output["retrieval"].get("mrr") is not None:
        output["mean_score"] = output["retrieval"]["mrr"]
    else:
        output["mean_score"] = 0.0

    output["n_samples"] = len(results)
    return output


# ─── comparison table ─────────────────────────────────────────

def compare_configs(results_paths: dict, run_judge: bool = True) -> dict:
    """
    Args:
        results_paths: {"BM25 baseline": "path1.json", ...}
        run_judge:     Whether to run Groq judge (costs tokens).

    Returns:
        {label: metrics_dict}
    """
    all_scores = {}

    for label, path in results_paths.items():
        with open(path) as f:
            results = json.load(f)
        scores = evaluate_results(results, label=label, run_judge=run_judge)
        all_scores[label] = scores

    # ── retrieval table ──
    print(f"\n{'='*75}")
    print("RETRIEVAL METRICS")
    print(f"{'='*75}")
    print(f"{'Config':<28} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>7} {'nDCG@5':>8}")
    print(f"{'-'*75}")
    for label, s in all_scores.items():
        r = s.get("retrieval", {})
        if r:
            print(f"{label:<28} {r.get('hit@1',0):>6.4f} {r.get('hit@3',0):>6.4f} "
                  f"{r.get('hit@5',0):>6.4f} {r.get('mrr',0):>7.4f} {r.get('ndcg@5',0):>8.4f}")

    # ── judge table ──
    if run_judge:
        print(f"\n{'='*75}")
        print("JUDGE METRICS (Groq-as-judge, 1-5 scale)")
        print(f"{'='*75}")
        print(f"{'Config':<28} {'Faith':>7} {'Relev':>7} {'Compl':>7} {'Mean':>7}")
        print(f"{'-'*75}")
        for label, s in all_scores.items():
            j = s.get("judge", {})
            if j:
                print(f"{label:<28} {j.get('faithfulness',0):>7.4f} {j.get('relevance',0):>7.4f} "
                      f"{j.get('completeness',0):>7.4f} {j.get('mean_score',0):>7.4f}")

    # ── improvement over baseline ──
    configs = list(all_scores.values())
    if len(configs) >= 2:
        baseline = configs[0]
        best     = configs[-1]
        b_score  = baseline.get("mean_score", 0)
        t_score  = best.get("mean_score", 0)
        if b_score > 0:
            improvement = ((t_score - b_score) / b_score) * 100
            direction   = "+" if improvement >= 0 else ""
            print(f"\nImprovement {baseline['label']} → {best['label']}: "
                  f"{direction}{improvement:.1f}% mean score")
        else:
            print("\nBaseline mean_score is 0 — improvement not computed.")

    return all_scores


def save_scores(scores: dict, path: str = "indexes/eval_scores.json"):
    existing = {}
    p = Path(path)
    if p.exists():
        with open(p) as f:
            existing = json.load(f)

    existing[scores.get("label", "run")] = scores

    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Scores saved → {path}")


# ─── entrypoint ───────────────────────────────────────────────

if __name__ == "__main__":
    with open("indexes/pipeline_results.json") as f:
        results = json.load(f)

    scores = evaluate_results(results, label="hybrid_hyde_v1", run_judge=True)
    save_scores(scores)