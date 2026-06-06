"""
generation_metrics.py
──────────────────────────────────────────────────────────────────────────────
LLM-as-judge metrics for evaluating the answer generation stage.
Completely independent of retrieval — fixes retrieved chunks as oracle input.

Metrics:
  1. Faithfulness      — Does the answer use only information from context?
  2. Answer Relevancy  — Does the answer actually address the question?
  3. Context Precision — Do the retrieved chunks contribute to the answer?
  4. Completeness      — Does the answer cover all required aspects?

These map closely to RAGAS but are implemented directly here so you can
audit, modify, and understand the scoring logic.

Usage:
    from generation_metrics import GenerationEvaluator

    eval = GenerationEvaluator()
    result = eval.evaluate_single(
        question="What regularization technique was used?",
        answer="The authors used dropout with p=0.1 on all attention layers.",
        context_chunks=["...chunk text 1...", "...chunk text 2..."],
        gold_answer="Dropout regularization was applied."  # optional
    )
    print(result)
"""

import json
import re
import time
from dataclasses import dataclass
import requests

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"


# ─────────────────────────── Data Structures ──────────────────────────────────

@dataclass
class GenerationScore:
    faithfulness: float          # 0-1: answer grounded in context
    answer_relevancy: float      # 0-1: answer addresses the question
    context_precision: float     # 0-1: context contains answer evidence
    completeness: float          # 0-1: answer covers what gold answer does
    has_gold: bool = False       # whether gold answer was provided

    def mean(self) -> float:
        scores = [self.faithfulness, self.answer_relevancy, self.context_precision]
        if self.has_gold:
            scores.append(self.completeness)
        return sum(scores) / len(scores)

    def __str__(self) -> str:
        lines = [
            f"  Faithfulness:     {self.faithfulness:.3f}  (grounded in retrieved chunks)",
            f"  Answer Relevancy: {self.answer_relevancy:.3f}  (addresses the question)",
            f"  Context Precision:{self.context_precision:.3f}  (retrieved chunks are useful)",
        ]
        if self.has_gold:
            lines.append(f"  Completeness:     {self.completeness:.3f}  (vs gold answer)")
        lines.append(f"  Mean:             {self.mean():.3f}")
        return "\n".join(lines)


# ─────────────────────────── Evaluator ────────────────────────────────────────

class GenerationEvaluator:
    """
    Evaluates generation quality using an LLM judge.

    All prompts are structured for reproducibility:
    - They request numeric scores (not text verdicts)
    - They include chain-of-thought to reduce anchoring bias
    - They are consistent enough to compare across methods
    """

    FAITHFULNESS_PROMPT = """You are evaluating whether an AI system's answer is faithful to the provided context.

Context chunks:
{context}

Question: {question}

Answer to evaluate: {answer}

Task: For each claim in the answer, determine if it is supported by the context.

Respond ONLY in JSON:
{{
  "claims_in_answer": ["claim 1", "claim 2"],
  "claims_supported": ["claim 1"],
  "faithfulness_score": 0.0,
  "reasoning": "brief explanation"
}}

faithfulness_score = supported_claims / total_claims (0.0 to 1.0)
If the answer makes no factual claims, score = 1.0"""

    RELEVANCY_PROMPT = """You are evaluating whether an answer addresses the question.

Question: {question}
Answer: {answer}

Rate on a 0–1 scale:
1.0 = directly and completely answers the question
0.5 = partially answers or addresses related but not core aspect
0.0 = does not answer the question at all

Respond ONLY in JSON:
{{
  "question_intent": "what the question is asking for",
  "answer_coverage": "what aspects of the question the answer covers",
  "relevancy_score": 0.0,
  "reasoning": "brief explanation"
}}"""

    CONTEXT_PRECISION_PROMPT = """You are evaluating whether retrieved context chunks are useful for answering a question.

Question: {question}
Answer: {answer}

Retrieved chunks (evaluate each):
{chunks_with_ids}

For each chunk, decide: does it contribute necessary information to answer this question?

Respond ONLY in JSON:
{{
  "chunk_usefulness": [
    {{"chunk_id": "0", "useful": true, "reason": "contains the statistical test name"}},
    {{"chunk_id": "1", "useful": false, "reason": "irrelevant background"}}
  ],
  "context_precision_score": 0.0,
  "reasoning": "brief explanation"
}}

context_precision_score = useful_chunks / total_chunks"""

    COMPLETENESS_PROMPT = """You are evaluating whether an AI answer covers the same information as a reference answer.

Question: {question}
Reference answer: {gold_answer}
AI answer: {answer}

Rate completeness:
1.0 = AI answer covers all key information in reference answer
0.5 = covers main point but misses important details  
0.0 = misses the key answer entirely

Respond ONLY in JSON:
{{
  "key_points_in_reference": ["point 1", "point 2"],
  "key_points_covered": ["point 1"],
  "completeness_score": 0.0,
  "reasoning": "brief explanation"
}}"""

    def __init__(self):
        pass  # no client needed — uses local Ollama

    def _call(self, prompt: str) -> dict:
        """Call local Ollama and parse JSON response."""
        for attempt in range(3):
            try:
                resp = requests.post(
                    OLLAMA_URL,
                    json={
                        "model":    OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream":   False,
                        "options":  {"temperature": 0.0, "num_predict": 800},
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                raw = resp.json()["message"]["content"].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                start = raw.find("{")
                end   = raw.rfind("}") + 1
                if start == -1 or end == 0:
                    raise ValueError(f"No JSON object in response: {raw[:200]!r}")
                return json.loads(raw[start:end])
            except Exception as e:
                if attempt == 2:
                    print(f"  LLM call failed: {e}")
                    return {}
                time.sleep(1)
        return {}

    def _score_faithfulness(self, question: str, answer: str, context: str) -> float:
        prompt = self.FAITHFULNESS_PROMPT.format(
            context=context[:3000],
            question=question,
            answer=answer
        )
        result = self._call(prompt)
        return float(result.get("faithfulness_score", 0.5))

    def _score_relevancy(self, question: str, answer: str) -> float:
        prompt = self.RELEVANCY_PROMPT.format(question=question, answer=answer)
        result = self._call(prompt)
        return float(result.get("relevancy_score", 0.5))

    def _score_context_precision(self, question: str, answer: str, chunks: list[str]) -> float:
        chunks_with_ids = "\n\n".join(
            f"[Chunk {i}]\n{chunk[:500]}"
            for i, chunk in enumerate(chunks)
        )
        prompt = self.CONTEXT_PRECISION_PROMPT.format(
            question=question,
            answer=answer,
            chunks_with_ids=chunks_with_ids
        )
        result = self._call(prompt)
        return float(result.get("context_precision_score", 0.5))

    def _score_completeness(self, question: str, answer: str, gold: str) -> float:
        prompt = self.COMPLETENESS_PROMPT.format(
            question=question,
            gold_answer=gold,
            answer=answer
        )
        result = self._call(prompt)
        return float(result.get("completeness_score", 0.5))

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context_chunks: list[str],
        gold_answer: str | None = None
    ) -> GenerationScore:
        """Evaluate one (question, answer, context) triple."""
        context_str = "\n\n---\n\n".join(context_chunks)

        faithfulness = self._score_faithfulness(question, answer, context_str)
        relevancy = self._score_relevancy(question, answer)
        precision = self._score_context_precision(question, answer, context_chunks)

        completeness = 0.0
        has_gold = gold_answer is not None
        if has_gold:
            completeness = self._score_completeness(question, answer, gold_answer)

        return GenerationScore(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            completeness=completeness,
            has_gold=has_gold
        )

    def evaluate_batch(
        self,
        qa_triples: list[dict],  # [{"question", "answer", "context_chunks", "gold_answer"?}]
        verbose: bool = True
    ) -> dict:
        """
        Evaluate a batch of (question, answer, context) triples.
        Returns aggregate statistics.
        """
        scores = []
        for i, item in enumerate(qa_triples):
            if verbose:
                print(f"  Evaluating {i+1}/{len(qa_triples)}: {item['question'][:50]}...")

            score = self.evaluate_single(
                question=item["question"],
                answer=item["answer"],
                context_chunks=item["context_chunks"],
                gold_answer=item.get("gold_answer")
            )
            scores.append(score)
            # no sleep needed — local Ollama has no rate limits

        if not scores:
            return {}

        n = len(scores)
        return {
            "n": n,
            "faithfulness": {
                "mean": sum(s.faithfulness for s in scores) / n,
                "min": min(s.faithfulness for s in scores),
                "max": max(s.faithfulness for s in scores),
            },
            "answer_relevancy": {
                "mean": sum(s.answer_relevancy for s in scores) / n,
            },
            "context_precision": {
                "mean": sum(s.context_precision for s in scores) / n,
            },
            "completeness": {
                "mean": sum(s.completeness for s in scores) / n
            } if scores[0].has_gold else None,
            "overall_mean": sum(s.mean() for s in scores) / n,
        }


# ─────────────────────────── Generation vs Retrieval Ablation ─────────────────

class RetrievalGenerationDecoupler:
    """
    Runs generation evaluation with FIXED oracle context (gold chunks).
    This isolates generation quality from retrieval quality.

    If generation is poor even with oracle context:
      → Problem is in the LLM / prompt
    If generation is good with oracle but poor with retrieved:
      → Problem is in retrieval
    """

    def __init__(self, generator_fn, gen_evaluator: GenerationEvaluator):
        """
        generator_fn: callable(question: str, context_chunks: list[str]) -> str
        gen_evaluator: GenerationEvaluator instance
        """
        self.generator_fn = generator_fn
        self.evaluator = gen_evaluator

    def evaluate_with_oracle(
        self,
        benchmark,              # Benchmark object
        chunk_lookup: dict[str, str],  # chunk_id → chunk_text
    ) -> dict:
        """
        Evaluate generation using gold chunks as context.
        Measures: can the LLM generate good answers IF retrieval was perfect?
        """
        print("\n[Oracle Context Evaluation — Isolates Generation Quality]")
        triples = []

        for qa in benchmark.qa_pairs:
            gold_chunks = [chunk_lookup[cid] for cid in qa.source_chunks if cid in chunk_lookup]
            if not gold_chunks:
                continue

            answer = self.generator_fn(qa.question, gold_chunks)
            triples.append({
                "question": qa.question,
                "answer": answer,
                "context_chunks": gold_chunks,
                "gold_answer": qa.answer
            })

        return self.evaluator.evaluate_batch(triples)

    def evaluate_with_retriever(
        self,
        benchmark,
        retriever_fn,           # question → list[chunk_id]
        chunk_lookup: dict[str, str],
        k: int = 5
    ) -> dict:
        """
        Evaluate generation using retrieved chunks (realistic setup).
        Compare to oracle results to isolate retrieval failures.
        """
        print(f"\n[Retriever Context Evaluation — K={k}]")
        triples = []

        for qa in benchmark.qa_pairs:
            retrieved_ids = retriever_fn(qa.question)[:k]
            retrieved_chunks = [chunk_lookup[cid] for cid in retrieved_ids if cid in chunk_lookup]
            if not retrieved_chunks:
                continue

            answer = self.generator_fn(qa.question, retrieved_chunks)
            triples.append({
                "question": qa.question,
                "answer": answer,
                "context_chunks": retrieved_chunks,
                "gold_answer": qa.answer
            })

        return self.evaluator.evaluate_batch(triples)