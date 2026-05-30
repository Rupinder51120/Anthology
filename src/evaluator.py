import json
import os
import numpy as np
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def run_ragas_eval(results: list[dict], label: str = "pipeline") -> dict:
    valid = [r for r in results if r["answer"] != "ERROR" and r["contexts"]]

    if not valid:
        print("No valid results to evaluate.")
        return {}

    data = {
        "question": [r["question"] for r in valid],
        "answer": [r["answer"] for r in valid],
        "contexts": [r["contexts"] for r in valid],
        "ground_truth": [r["ground_truth"] for r in valid],
    }

    dataset = Dataset.from_dict(data)

    print(f"\nRunning RAGAS eval on {len(valid)} samples [{label}]...")

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )

    result = {
        "label": label,
        "n_samples": len(valid),
        "faithfulness": round(float(scores["faithfulness"]), 4),
        "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
        "context_precision": round(float(scores["context_precision"]), 4),
        "mean_score": round(float(np.mean([
            scores["faithfulness"],
            scores["answer_relevancy"],
            scores["context_precision"]
        ])), 4)
    }

    print(f"\n{'='*45}")
    print(f"RAGAS Results — {label}")
    print(f"{'='*45}")
    print(f"Faithfulness:      {result['faithfulness']:.4f}")
    print(f"Answer relevancy:  {result['answer_relevancy']:.4f}")
    print(f"Context precision: {result['context_precision']:.4f}")
    print(f"Mean score:        {result['mean_score']:.4f}")
    print(f"{'='*45}")

    return result


def compare_configs(results_paths: dict) -> dict:
    """
    results_paths: {"BM25 baseline": "path1.json", "Hybrid+HyDE": "path2.json"}
    Returns comparison table for your resume numbers.
    """
    all_scores = {}

    for label, path in results_paths.items():
        with open(path) as f:
            results = json.load(f)
        scores = run_ragas_eval(results, label=label)
        all_scores[label] = scores

    print(f"\n{'='*55}")
    print("COMPARISON TABLE")
    print(f"{'='*55}")
    print(f"{'Config':<25} {'Faith':>8} {'Relev':>8} {'Prec':>8} {'Mean':>8}")
    print(f"{'-'*55}")

    for label, scores in all_scores.items():
        if scores:
            print(f"{label:<25} {scores['faithfulness']:>8.4f} {scores['answer_relevancy']:>8.4f} {scores['context_precision']:>8.4f} {scores['mean_score']:>8.4f}")

    # compute improvement from first to last
    configs = list(all_scores.values())
    if len(configs) >= 2:
        baseline = configs[0]
        best = configs[-1]
        if baseline and best:
            improvement = ((best["mean_score"] - baseline["mean_score"]) / baseline["mean_score"]) * 100
            print(f"\nImprovement over baseline: +{improvement:.1f}% mean score")
            print(f"This is your resume number.")

    return all_scores


def save_scores(scores: dict, path: str = "indexes/eval_scores.json"):
    existing = {}
    if Path(path).exists():
        with open(path) as f:
            existing = json.load(f)

    existing[scores.get("label", "run")] = scores

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Scores saved → {path}")


if __name__ == "__main__":
    # run single eval
    with open("indexes/pipeline_results.json") as f:
        results = json.load(f)

    scores = run_ragas_eval(results, label="hybrid_hyde_v1")
    save_scores(scores)