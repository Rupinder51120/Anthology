"""
src/evaluator.py

Migrated from Groq-as-judge to Ollama-as-judge.
- No API key needed
- No rate limits  
- No sleep needed between calls
- Retrieval metrics unchanged (no LLM needed)
"""

import json
import time
from pathlib import Path

import numpy as np
import requests

OLLAMA_URL   = "http://localhost:11434/api/generate"
JUDGE_MODEL  = "qwen2.5:7b"


# ─── retrieval metrics ────────────────────────────────────────

def _hit_at_k(sources: list[str], expected_source: str, k: int) -> int:
    return int(any(expected_source in s for s in sources[:k]))


def _reciprocal_rank(sources: list[str], expected_source: str) -> float:
    for i, s in enumerate(sources):
        if expected_source in s:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(sources: list[str], expected_source: str, k: int = 5) -> float:
    relevance = [1 if expected_source in s else 0 for s in sources[:k]]
    dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(relevance))
    idcg = 1.0
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(results: list[dict], k_values: list[int] = [1, 3, 5]) -> dict:
    valid = [
        r for r in results
        if r.get("source_chunk") and r.get("sources") and r["answer"] != "ERROR"
        and "Generation failed" not in r.get("answer", "")
    ]

    if not valid:
        print("  No valid results with source_chunk — retrieval metrics skipped.")
        return {}

    hit_scores  = {k: [] for k in k_values}
    mrr_scores  = []
    ndcg_scores = []

    for r in valid:
        expected = r["source_chunk"]
        sources  = r["sources"]
        for k in k_values:
            hit_scores[k].append(_hit_at_k(sources, expected, k))
        mrr_scores.append(_reciprocal_rank(sources, expected))
        ndcg_scores.append(_ndcg_at_k(sources, expected, k=5))

    metrics = {f"hit@{k}": round(float(np.mean(hit_scores[k])), 4) for k in k_values}
    metrics["mrr"]    = round(float(np.mean(mrr_scores)), 4)
    metrics["ndcg@5"] = round(float(np.mean(ndcg_scores)), 4)
    metrics["n_eval"] = len(valid)
    return metrics


# ─── Ollama-as-judge ─────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a RAG system answer. Respond ONLY with a JSON object.

Question: {question}
Expected answer: {ground_truth}
Retrieved context: {context}
System answer: {answer}

Score each dimension 1-5:
- faithfulness: grounded in context? (1=hallucinated, 5=fully grounded)
- relevance: answers the question? (1=off-topic, 5=direct answer)
- completeness: covers expected answer? (1=missing key points, 5=complete)

Respond ONLY with this JSON, nothing else:
{{"faithfulness": <int>, "relevance": <int>, "completeness": <int>}}"""


def _call_judge(question: str, ground_truth: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        context=context[:1500],
        answer=answer[:800],
    )
    resp = requests.post(
        OLLAMA_URL,
        json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.0, "num_predict": 80}},
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["response"].strip()
    # extract JSON from response
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in: {raw}")
    return json.loads(raw[start:end])


def compute_judge_metrics(results: list[dict]) -> dict:
    valid = [
        r for r in results
        if r["answer"] not in ("ERROR", "")
        and "Generation failed" not in r.get("answer", "")
        and r.get("contexts")
    ]

    if not valid:
        print("  No valid results for judge evaluation.")
        return {}

    scores = {"faithfulness": [], "relevance": [], "completeness": []}
    failed = 0

    for i, r in enumerate(valid):
        context = "\n---\n".join(r["contexts"][:3])
        try:
            s = _call_judge(
                question=r["question"],
                ground_truth=r["ground_truth"],
                context=context,
                answer=r["answer"],
            )
            for dim in scores:
                scores[dim].append(float(s.get(dim, 0)))
            if (i + 1) % 10 == 0:
                print(f"    Judge progress: {i+1}/{len(valid)}")
        except Exception as e:
            print(f"  Judge failed on question {i+1}: {e}")
            for dim in scores:
                scores[dim].append(0.0)
            failed += 1

    if not any(scores["faithfulness"]):
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
    print(f"\n{'='*50}")
    print(f"Evaluating: {label}")
    print(f"{'='*50}")

    output = {"label": label}

    print("  Computing retrieval metrics...")
    retrieval = compute_retrieval_metrics(results)
    output["retrieval"] = retrieval
    if retrieval:
        print(f"  Hit@1={retrieval.get('hit@1'):.4f}  Hit@3={retrieval.get('hit@3'):.4f}  "
              f"Hit@5={retrieval.get('hit@5'):.4f}  MRR={retrieval.get('mrr'):.4f}  "
              f"nDCG@5={retrieval.get('ndcg@5'):.4f}  (n={retrieval.get('n_eval')})")

    if run_judge:
        print("  Running Ollama-as-judge...")
        judge = compute_judge_metrics(results)
        output["judge"] = judge
        if judge:
            print(f"  Faithfulness={judge['faithfulness']:.4f}  "
                  f"Relevance={judge['relevance']:.4f}  "
                  f"Completeness={judge['completeness']:.4f}  "
                  f"Mean={judge['mean_score']:.4f}  (n={judge['n_eval']})")
    else:
        output["judge"] = {}

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
    all_scores = {}

    for label, path in results_paths.items():
        with open(path) as f:
            results = json.load(f)
        scores = evaluate_results(results, label=label, run_judge=run_judge)
        all_scores[label] = scores

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

    if run_judge:
        print(f"\n{'='*75}")
        print("JUDGE METRICS (Ollama qwen2.5:7b, 1-5 scale)")
        print(f"{'='*75}")
        print(f"{'Config':<28} {'Faith':>7} {'Relev':>7} {'Compl':>7} {'Mean':>7}")
        print(f"{'-'*75}")
        for label, s in all_scores.items():
            j = s.get("judge", {})
            if j:
                print(f"{label:<28} {j.get('faithfulness',0):>7.4f} {j.get('relevance',0):>7.4f} "
                      f"{j.get('completeness',0):>7.4f} {j.get('mean_score',0):>7.4f}")

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


if __name__ == "__main__":
    with open("indexes/pipeline_results.json") as f:
        results = json.load(f)
    scores = evaluate_results(results, label="ollama_test", run_judge=True)
    save_scores(scores)