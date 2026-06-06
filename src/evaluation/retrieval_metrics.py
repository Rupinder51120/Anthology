"""
retrieval_metrics.py
──────────────────────────────────────────────────────────────────────────────
Pure-Python retrieval evaluation metrics.

All metrics operate on ranked lists of retrieved item IDs versus
a set of gold (relevant) item IDs. No retrieval logic here — just math.

Two granularities:
  - Chunk-level:  gold_ids are chunk IDs   (fine-grained)
  - Paper-level:  gold_ids are paper names (your current setup)

Usage:
    from retrieval_metrics import RetrievalMetrics, RetrievalEvaluator

    metrics = RetrievalMetrics()
    score = metrics.recall_at_k(retrieved=["c1","c2","c3"], gold={"c2","c5"}, k=3)
    # → 0.5
"""

import math
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict


# ─────────────────────────── Core Metric Functions ────────────────────────────

class RetrievalMetrics:
    """
    Stateless metric computations.
    retrieved: ordered list of IDs (rank 1 first)
    gold:      set of relevant IDs
    k:         cutoff
    """

    def recall_at_k(self, retrieved: list[str], gold: set[str], k: int) -> float:
        """
        Fraction of gold items found in top-k retrieved.
        If |gold| == 0, returns 1.0 (vacuously true).

        Use this when: you need ALL relevant evidence (mentor system).
        """
        if not gold:
            return 1.0
        top_k = set(retrieved[:k])
        return len(top_k & gold) / len(gold)

    def hit_at_k(self, retrieved: list[str], gold: set[str], k: int) -> float:
        """
        Binary: 1.0 if ANY gold item is in top-k, else 0.0.
        Use this when: you only need one good chunk to answer.
        """
        top_k = set(retrieved[:k])
        return 1.0 if top_k & gold else 0.0

    def precision_at_k(self, retrieved: list[str], gold: set[str], k: int) -> float:
        """
        Fraction of top-k that are relevant.
        Use this when: you want to minimize noise sent to the LLM generator.
        """
        if k == 0:
            return 0.0
        top_k = retrieved[:k]
        relevant = sum(1 for r in top_k if r in gold)
        return relevant / k

    def mrr(self, retrieved: list[str], gold: set[str]) -> float:
        """
        Mean Reciprocal Rank: 1/rank_of_first_relevant_item.
        Use this when: you care about ranking quality, not coverage.
        """
        for rank, item in enumerate(retrieved, start=1):
            if item in gold:
                return 1.0 / rank
        return 0.0

    def average_precision(self, retrieved: list[str], gold: set[str]) -> float:
        """
        Area under the precision-recall curve (AP).
        Use this when: you want to reward both early and complete retrieval.
        """
        if not gold:
            return 0.0
        hits = 0
        sum_precision = 0.0
        for rank, item in enumerate(retrieved, start=1):
            if item in gold:
                hits += 1
                sum_precision += hits / rank
        return sum_precision / len(gold)

    def ndcg_at_k(
        self,
        retrieved: list[str],
        gold: set[str],
        k: int,
        relevance_scores: dict[str, float] | None = None
    ) -> float:
        """
        Normalized Discounted Cumulative Gain.
        Supports graded relevance if relevance_scores is provided
        (e.g., {"chunk_1": 2.0, "chunk_2": 1.0} for strong/weak relevance).
        Falls back to binary relevance if None.
        """
        def rel(item: str) -> float:
            if relevance_scores:
                return relevance_scores.get(item, 0.0)
            return 1.0 if item in gold else 0.0

        def dcg(items: list[str], k: int) -> float:
            return sum(
                rel(item) / math.log2(rank + 1)
                for rank, item in enumerate(items[:k], start=1)
            )

        actual_dcg = dcg(retrieved, k)
        ideal_items = sorted(
            (relevance_scores or {i: 1.0 for i in gold}).keys(),
            key=lambda x: (relevance_scores or {}).get(x, 1.0),
            reverse=True
        )
        ideal_dcg = dcg(ideal_items, k)

        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def paper_level_hit(
        self,
        retrieved_chunks: list[str],
        gold_paper: str,
        chunk_to_paper: dict[str, str],
        k: int
    ) -> float:
        """
        Your current metric: did any top-k chunk come from the gold paper?
        chunk_to_paper maps chunk_id → paper_name.
        """
        top_k = retrieved_chunks[:k]
        for chunk_id in top_k:
            if chunk_to_paper.get(chunk_id) == gold_paper:
                return 1.0
        return 0.0


# ─────────────────────────── Evaluator (Runs Full Benchmark) ──────────────────

@dataclass
class EvalResult:
    method_name: str
    k_values: list[int]
    # Per-K metrics
    recall: dict[int, float] = field(default_factory=dict)
    hit: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    # Overall metrics
    mrr: float = 0.0
    map_score: float = 0.0
    # Breakdown by question type
    by_type: dict[str, dict] = field(default_factory=dict)
    # Breakdown by difficulty
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    n_questions: int = 0

    def summary(self) -> str:
        lines = [f"\n{'='*55}", f"  {self.method_name}", f"{'='*55}"]
        lines.append(f"  Questions evaluated: {self.n_questions}")
        lines.append(f"  MRR:                 {self.mrr:.4f}")
        lines.append(f"  MAP:                 {self.map_score:.4f}")
        for k in self.k_values:
            lines.append(f"\n  K = {k}:")
            lines.append(f"    Recall@{k}:     {self.recall.get(k, 0):.4f}")
            lines.append(f"    Hit@{k}:        {self.hit.get(k, 0):.4f}")
            lines.append(f"    Precision@{k}:  {self.precision.get(k, 0):.4f}")
            lines.append(f"    NDCG@{k}:       {self.ndcg.get(k, 0):.4f}")
        if self.by_type:
            lines.append(f"\n  Breakdown by question type:")
            for qtype, scores in self.by_type.items():
                lines.append(f"    {qtype:15s}  Recall@5={scores.get('recall@5', 0):.3f}  Hit@5={scores.get('hit@5', 0):.3f}  n={scores.get('n', 0)}")
        return "\n".join(lines)


class RetrievalEvaluator:
    """
    Evaluates a retrieval method against a benchmark.

    Usage:
        evaluator = RetrievalEvaluator(k_values=[1, 5, 10])

        # retriever_fn: callable(question: str) -> list[str]  (ordered chunk ids)
        result = evaluator.evaluate(
            method_name="BM25",
            retriever_fn=bm25_retrieve,
            benchmark=my_benchmark,
            chunk_to_paper=chunk_paper_map   # for paper-level eval
        )
        print(result.summary())
    """

    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [1, 5, 10]
        self.metrics = RetrievalMetrics()

    def evaluate(
        self,
        method_name: str,
        retriever_fn,           # callable: question → list[chunk_id]
        benchmark,              # Benchmark object from benchmark_generator
        chunk_to_paper: dict[str, str] | None = None,
        granularity: str = "chunk",  # "chunk" or "paper"
        verbose: bool = False,
    ) -> EvalResult:
        """
        granularity="chunk":  gold = qa.source_chunks (requires chunk IDs)
        granularity="paper":  gold = {qa.source_paper} (your current setup)
        """
        result = EvalResult(
            method_name=method_name,
            k_values=self.k_values,
            n_questions=len(benchmark.qa_pairs)
        )

        # Accumulators
        recall_sums = defaultdict(float)
        hit_sums = defaultdict(float)
        precision_sums = defaultdict(float)
        ndcg_sums = defaultdict(float)
        mrr_sum = 0.0
        ap_sum = 0.0

        # Per-type / per-difficulty accumulators
        type_acc: dict[str, dict] = defaultdict(lambda: defaultdict(float))
        type_count: dict[str, int] = defaultdict(int)
        diff_acc: dict[str, dict] = defaultdict(lambda: defaultdict(float))
        diff_count: dict[str, int] = defaultdict(int)

        for qa in benchmark.qa_pairs:
            retrieved = retriever_fn(qa.question)

            # Build gold set
            if granularity == "paper":
                # Map: if chunk_to_paper is provided, translate retrieved chunk IDs to papers
                if chunk_to_paper:
                    retrieved_papers = [chunk_to_paper.get(c, c) for c in retrieved]
                else:
                    retrieved_papers = retrieved
                gold = {qa.source_paper}
                eval_retrieved = retrieved_papers
            else:
                gold = set(qa.source_chunks)
                eval_retrieved = retrieved

            if verbose:
                print(f"\n  Q: {qa.question[:60]}...")
                print(f"  Gold: {gold}")
                print(f"  Top-3 retrieved: {eval_retrieved[:3]}")

            mrr_val = self.metrics.mrr(eval_retrieved, gold)
            ap_val = self.metrics.average_precision(eval_retrieved, gold)
            mrr_sum += mrr_val
            ap_sum += ap_val

            qtype = qa.question_type or "unknown"
            diff = qa.difficulty or "unknown"
            type_count[qtype] += 1
            diff_count[diff] += 1

            for k in self.k_values:
                r = self.metrics.recall_at_k(eval_retrieved, gold, k)
                h = self.metrics.hit_at_k(eval_retrieved, gold, k)
                p = self.metrics.precision_at_k(eval_retrieved, gold, k)
                n = self.metrics.ndcg_at_k(eval_retrieved, gold, k)

                recall_sums[k] += r
                hit_sums[k] += h
                precision_sums[k] += p
                ndcg_sums[k] += n

                type_acc[qtype][f"recall@{k}"] += r
                type_acc[qtype][f"hit@{k}"] += h
                diff_acc[diff][f"recall@{k}"] += r

        n = result.n_questions
        if n == 0:
            return result

        result.mrr = mrr_sum / n
        result.map_score = ap_sum / n

        for k in self.k_values:
            result.recall[k] = recall_sums[k] / n
            result.hit[k] = hit_sums[k] / n
            result.precision[k] = precision_sums[k] / n
            result.ndcg[k] = ndcg_sums[k] / n

        for qtype, cnt in type_count.items():
            result.by_type[qtype] = {
                metric: val / cnt
                for metric, val in type_acc[qtype].items()
            }
            result.by_type[qtype]["n"] = cnt

        for diff, cnt in diff_count.items():
            result.by_difficulty[diff] = {
                metric: val / cnt
                for metric, val in diff_acc[diff].items()
            }
            result.by_difficulty[diff]["n"] = cnt

        return result


# ─────────────────────────── Multi-Method Comparison ──────────────────────────

class RetrieverComparison:
    """
    Compare multiple retrieval methods on the same benchmark.

    Usage:
        cmp = RetrieverComparison(k_values=[1, 5, 10])
        cmp.add_method("BM25", bm25_fn)
        cmp.add_method("Hybrid", hybrid_fn)
        cmp.add_method("Hybrid+Rerank", rerank_fn)
        results = cmp.run(benchmark, chunk_to_paper)
        cmp.print_comparison_table(results)
        cmp.export_results(results, "./eval_results.json")
    """

    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [1, 5, 10]
        self.evaluator = RetrievalEvaluator(k_values=self.k_values)
        self.methods: list[tuple[str, Any]] = []

    def add_method(self, name: str, retriever_fn):
        self.methods.append((name, retriever_fn))

    def run(self, benchmark, chunk_to_paper: dict | None = None, granularity: str = "chunk") -> list[EvalResult]:
        results = []
        for name, fn in self.methods:
            print(f"\nEvaluating: {name}")
            result = self.evaluator.evaluate(
                method_name=name,
                retriever_fn=fn,
                benchmark=benchmark,
                chunk_to_paper=chunk_to_paper,
                granularity=granularity
            )
            results.append(result)
            print(result.summary())
        return results

    def print_comparison_table(self, results: list[EvalResult], k: int = 5):
        """Print a side-by-side comparison table for a given K."""
        print(f"\n{'='*70}")
        print(f"  COMPARISON TABLE  @K={k}")
        print(f"{'='*70}")
        header = f"{'Method':25s}  {'Recall@'+str(k):12s}  {'Hit@'+str(k):10s}  {'Precision@'+str(k):14s}  MRR"
        print(header)
        print("-" * 70)
        for r in sorted(results, key=lambda x: -x.recall.get(k, 0)):
            print(
                f"{r.method_name:25s}  "
                f"{r.recall.get(k, 0):.4f}        "
                f"{r.hit.get(k, 0):.4f}      "
                f"{r.precision.get(k, 0):.4f}            "
                f"{r.mrr:.4f}"
            )
        print("="*70)

    def export_results(self, results: list[EvalResult], path: str):
        import json
        from dataclasses import asdict
        output = []
        for r in results:
            d = asdict(r)
            # Convert int keys in dicts to strings for JSON serialization
            for field_name in ["recall", "hit", "precision", "ndcg"]:
                d[field_name] = {str(k): v for k, v in d[field_name].items()}
            output.append(d)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults exported to {path}")