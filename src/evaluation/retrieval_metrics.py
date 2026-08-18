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
    from retrieval_metrics import RetrievalMetrics

    metrics = RetrievalMetrics()
    score = metrics.recall_at_k(retrieved=["c1","c2","c3"], gold={"c2","c5"}, k=3)
    # → 0.5
"""

import math


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

