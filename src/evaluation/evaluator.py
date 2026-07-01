"""
src/evaluation/evaluator.py
Groq-as-judge (llama-3.1-8b-instant) — no Ollama needed, uses existing GROQ_API_KEY
Metrics: Hit@k, MRR, nDCG@5 (retrieval) + Faithfulness, Relevance, Completeness (generation)
These are RAGAS-equivalent metrics implemented directly.

FIX (audit: evaluation flaw): retrieval metrics previously only checked
paper-level relevance ("did a chunk from the right paper show up"), even
though the field was misleadingly named source_chunk. Retrieving the WRONG
chunk from the right paper counted as a perfect hit. Paper-level metrics
are KEPT (still meaningful — measures "did retrieval find the right
document at all"), and chunk-level metrics are ADDED alongside them,
activating automatically when a result has a source_chunk_id field
(backward compatible — old datasets/results without that field just skip
chunk-level scoring, no errors).
"""

import json
import os
from pathlib import Path

import numpy as np
from groq import Groq
from api.core.models import GROQ_CHAT_MODEL
from api.core.config import get_settings

settings = get_settings()
JUDGE_MODEL = GROQ_CHAT_MODEL


def _get_client():
    return Groq(api_key=settings.groq_api_key.get_secret_value())


def _norm(s: str) -> str:
    return Path(s).stem if s else ""


# ── Paper-level retrieval metrics (existing, unchanged logic) ──

def _hit_at_k(sources, expected, k):
    if not sources or not expected:
        return 0
    exp = _norm(expected)
    return int(any(_norm(s) == exp for s in sources[:k]))

def _reciprocal_rank(sources, expected):
    if not sources or not expected:
        return 0.0
    exp = _norm(expected)
    for i, s in enumerate(sources):
        if _norm(s) == exp:
            return 1.0 / (i + 1)
    return 0.0

def _ndcg_at_k(sources, expected, k=5):
    if not sources or not expected:
        return 0.0
    k = min(k, len(sources))
    exp = _norm(expected)
    relevance = [1 if _norm(s) == exp else 0 for s in sources[:k]]
    dcg  = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted(relevance, reverse=True)))
    return (dcg / idcg) if idcg > 0.0 else 0.0


# ── Chunk-level retrieval metrics (NEW — exact chunk_id match, no stemming) ──

def _hit_at_k_chunk(chunk_ids, expected_chunk_id, k):
    if not chunk_ids or not expected_chunk_id:
        return 0
    return int(expected_chunk_id in chunk_ids[:k])

def _reciprocal_rank_chunk(chunk_ids, expected_chunk_id):
    if not chunk_ids or not expected_chunk_id:
        return 0.0
    for i, cid in enumerate(chunk_ids):
        if cid == expected_chunk_id:
            return 1.0 / (i + 1)
    return 0.0

def _ndcg_at_k_chunk(chunk_ids, expected_chunk_id, k=5):
    if not chunk_ids or not expected_chunk_id:
        return 0.0
    k = min(k, len(chunk_ids))
    relevance = [1 if cid == expected_chunk_id else 0 for cid in chunk_ids[:k]]
    dcg  = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted(relevance, reverse=True)))
    return (dcg / idcg) if idcg > 0.0 else 0.0


def compute_retrieval_metrics(results, k_values=[1, 3, 5]):
    """Paper-level metrics (existing behavior, unchanged)."""
    valid = [
        r for r in results
        if r.get("source_chunk") and r.get("sources")
        and r["answer"] not in ("ERROR", "")
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


def compute_chunk_retrieval_metrics(results, k_values=[1, 3, 5]):
    """
    Chunk-level metrics (NEW). Requires results to have:
      - source_chunk_id: ground truth chunk_id the question was anchored to
      - chunk_ids: list of chunk_ids actually retrieved, in rank order

    Backward compatible: if results don't have these fields (older
    datasets/pipeline_results), returns {} instead of erroring, so existing
    callers that don't expect this can simply ignore an empty dict.
    """
    valid = [
        r for r in results
        if r.get("source_chunk_id") and r.get("chunk_ids")
        and r["answer"] not in ("ERROR", "")
        and "Generation failed" not in r.get("answer", "")
    ]
    if not valid:
        print("  No valid results with source_chunk_id — chunk-level metrics skipped "
              "(dataset/pipeline may predate this field; this is expected for old data).")
        return {}

    hit_scores  = {k: [] for k in k_values}
    mrr_scores  = []
    ndcg_scores = []

    for r in valid:
        expected  = r["source_chunk_id"]
        chunk_ids = r["chunk_ids"]
        for k in k_values:
            hit_scores[k].append(_hit_at_k_chunk(chunk_ids, expected, k))
        mrr_scores.append(_reciprocal_rank_chunk(chunk_ids, expected))
        ndcg_scores.append(_ndcg_at_k_chunk(chunk_ids, expected, k=5))

    metrics = {f"hit@{k}": round(float(np.mean(hit_scores[k])), 4) for k in k_values}
    metrics["mrr"]    = round(float(np.mean(mrr_scores)), 4)
    metrics["ndcg@5"] = round(float(np.mean(ndcg_scores)), 4)
    metrics["n_eval"] = len(valid)
    return metrics


# ── Groq-as-judge (RAGAS-equivalent) ─────────────────────────

FAITHFULNESS_PROMPT = """You are evaluating a RAG system. Respond ONLY with JSON.

Context:
{context}

Question: {question}
Answer: {answer}

List every factual claim in the answer. For each, is it supported by the context above?

Respond ONLY with this JSON:
{{"claims": ["claim1", "claim2"], "supported": ["claim1"], "faithfulness_score": 0.0}}

faithfulness_score = supported / total claims. If no claims, score = 1.0"""

RELEVANCY_PROMPT = """You are evaluating whether an answer addresses the question.

Question: {question}
Answer: {answer}

Score 0.0-1.0: 1.0=fully answers, 0.5=partially, 0.0=irrelevant.

Respond ONLY with this JSON:
{{"relevancy_score": 0.0, "reasoning": "one line"}}"""

COMPLETENESS_PROMPT = """You are evaluating answer completeness vs a reference.

Question: {question}
Reference: {ground_truth}
Answer: {answer}

Score 0.0-1.0: 1.0=covers all key points, 0.5=covers main point, 0.0=misses it.

Respond ONLY with this JSON:
{{"completeness_score": 0.0, "reasoning": "one line"}}"""


def _call_groq(prompt: str) -> dict:
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {}
        return json.loads(raw[start:end])
    except Exception as e:
        print(f"  Groq judge error: {e}")
        return {}


def compute_judge_metrics(results):
    valid = [
        r for r in results
        if r["answer"] not in ("ERROR", "")
        and "Generation failed" not in r.get("answer", "")
        and r.get("contexts")
    ]
    if not valid:
        print("  No valid results for judge evaluation.")
        return {}

    faith_scores, relev_scores, compl_scores = [], [], []
    failed = 0

    for i, r in enumerate(valid):
        context = "\n---\n".join(r["contexts"][:3])[:2000]
        try:
            f = _call_groq(FAITHFULNESS_PROMPT.format(
                context=context, question=r["question"], answer=r["answer"][:600]
            ))
            faith_scores.append(float(f.get("faithfulness_score", 0.5)))

            rv = _call_groq(RELEVANCY_PROMPT.format(
                question=r["question"], answer=r["answer"][:600]
            ))
            relev_scores.append(float(rv.get("relevancy_score", 0.5)))

            if r.get("ground_truth"):
                c = _call_groq(COMPLETENESS_PROMPT.format(
                    question=r["question"],
                    ground_truth=r["ground_truth"],
                    answer=r["answer"][:600]
                ))
                compl_scores.append(float(c.get("completeness_score", 0.5)))

            if (i + 1) % 5 == 0:
                print(f"    Judge progress: {i+1}/{len(valid)}")

        except Exception as e:
            print(f"  Judge failed on {i+1}: {e}")
            faith_scores.append(0.5)
            relev_scores.append(0.5)
            failed += 1

    if not faith_scores:
        return {}

    result = {
        "faithfulness":  round(float(np.mean(faith_scores)), 4),
        "relevance":     round(float(np.mean(relev_scores)), 4),
        "completeness":  round(float(np.mean(compl_scores)), 4) if compl_scores else None,
        "n_eval":        len(valid),
        "n_failed":      failed,
    }
    scores = [result["faithfulness"], result["relevance"]]
    if result["completeness"] is not None:
        scores.append(result["completeness"])
    result["mean_score"] = round(float(np.mean(scores)), 4)
    return result


# ── Combined eval ─────────────────────────────────────────────

def evaluate_results(results, label="pipeline", run_judge=True):
    print(f"\n{'='*50}\nEvaluating: {label}\n{'='*50}")
    output = {"label": label}

    print("  Computing paper-level retrieval metrics...")
    retrieval = compute_retrieval_metrics(results)
    output["retrieval"] = retrieval
    if retrieval:
        print(f"  Hit@1={retrieval['hit@1']}  Hit@3={retrieval['hit@3']}  "
              f"Hit@5={retrieval['hit@5']}  MRR={retrieval['mrr']}  "
              f"nDCG@5={retrieval['ndcg@5']}  (n={retrieval['n_eval']})")

    print("  Computing chunk-level retrieval metrics...")
    chunk_retrieval = compute_chunk_retrieval_metrics(results)
    output["chunk_retrieval"] = chunk_retrieval
    if chunk_retrieval:
        print(f"  [chunk] Hit@1={chunk_retrieval['hit@1']}  Hit@3={chunk_retrieval['hit@3']}  "
              f"Hit@5={chunk_retrieval['hit@5']}  MRR={chunk_retrieval['mrr']}  "
              f"nDCG@5={chunk_retrieval['ndcg@5']}  (n={chunk_retrieval['n_eval']})")

    if run_judge:
        print("  Running Groq-as-judge (RAGAS-equivalent)...")
        judge = compute_judge_metrics(results)
        output["judge"] = judge
        if judge:
            print(f"  Faithfulness={judge['faithfulness']}  "
                  f"Relevance={judge['relevance']}  "
                  f"Completeness={judge.get('completeness')}  "
                  f"Mean={judge['mean_score']}  (n={judge['n_eval']})")
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


def save_scores(scores, path="indexes/eval_scores.json"):
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