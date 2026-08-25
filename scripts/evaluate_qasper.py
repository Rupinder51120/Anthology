"""
scripts/evaluate_qasper.py

External retrieval-generalization evaluation against QASPER
(allenai/qasper, https://huggingface.co/datasets/allenai/qasper), run
alongside -- not instead of -- Anthology's existing internal 247-question
self-generated benchmark (benchmarks/qa_dataset_v1.json).

WHAT THIS IS
------------
QASPER is a real, third-party, CC-BY-4.0-licensed academic dataset: 5,049
questions over 1,585 NLP papers, each question written by a practitioner
who had only read the paper's title/abstract, answered by a second
practitioner with paragraph-level supporting evidence. Train (888 papers /
2,593 questions) and validation (281 papers / 1,005 questions) splits have
public answers; the test split's answers are leaderboard-only and are not
used here.

This script evaluates RETRIEVAL ONLY (Hit@1/Hit@5/MRR at the paper level
and at the evidence-paragraph level), using Anthology's actual SPECTER2
embedder (src/retrieval/embedder.py) for the dense strategy, rank_bm25 for
a sparse strategy, and Reciprocal Rank Fusion for a hybrid strategy --
mirroring 3 of the 6 strategies in the internal benchmark. It deliberately
does NOT evaluate generation/answer quality: that would require Groq/Cohere
API calls against ~1,000+ questions, real paid-API spend, and is out of
scope for a retrieval-generalization check. See LIMITATIONS below.

WHY IT DOESN'T TOUCH THE REAL CORPUS OR DB
-------------------------------------------
QASPER's 1,585 papers are NOT Anthology's ingested 121-paper corpus, and
full paper text (chunked by section into paragraphs) ships directly in the
dataset -- no PDF download/parsing is needed at all. This script builds a
completely separate, in-memory, per-run vector index from QASPER's own
paragraph text; it never writes to Postgres, never touches the `papers`/
`chunks` tables, and never modifies benchmarks/qa_dataset_v1.json or any
indexes/results_*.json artifact from the internal benchmark.

SCOPE OF THIS RUN
------------------
Per the operating instruction not to process a huge dataset blindly, this
script defaults to a bounded pilot sample (a fixed number of validation-
split papers, deterministically selected) rather than the full 281-paper /
1,005-question validation split, so a first run completes in a few minutes
and results can be sanity-checked before any decision to scale up. Use
--num-papers to change the sample size (up to 281, the full validation
split) and --split {validation,train} to choose the split.

LIMITATIONS (documented, not hidden)
-------------------------------------
- Retrieval only. No faithfulness/relevance/completeness or any generation
  quality is measured here.
- Paragraph-level chunking uses QASPER's own paragraph boundaries (the
  dataset's native evidence granularity) rather than Anthology's full PDF
  ingestion chunker -- there's no table/figure/OCR content to chunk here,
  so forcing paragraphs through the multimodal chunker would add complexity
  without changing what's being measured.
- Dense/sparse/hybrid_rrf only -- no Cohere reranking (avoids per-question
  paid API calls against a 1,000+ question set).
- A pilot-sized run (see SCOPE above) is a sample of the validation split,
  not the full split, unless explicitly re-run with a larger --num-papers.

USAGE
-----
    python scripts/evaluate_qasper.py --num-papers 40
    python scripts/evaluate_qasper.py --num-papers 281 --split validation  # full split

Results are saved to indexes/qasper_external_eval_<split>_<n>papers.json
(a new, clearly-named artifact -- never overwrites any existing internal
benchmark file).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.embedder import embed_texts  # noqa: E402


def _load_qasper_papers(split: str, num_papers: int):
    from datasets import load_dataset

    print(f"Loading QASPER ({split} split) from HuggingFace...")
    # allenai/qasper's default revision still ships a legacy loading script,
    # which recent `datasets` versions refuse to execute (security change).
    # The repo's auto-converted Parquet revision works with the modern,
    # script-free loading path.
    ds = load_dataset("allenai/qasper", split=split, revision="refs/convert/parquet")
    n = min(num_papers, len(ds))
    print(f"QASPER {split} split has {len(ds)} papers total; using first {n}.")
    return ds.select(range(n))


def _build_paragraph_corpus(papers) -> tuple[list[dict], dict]:
    """
    Flattens each paper's full_text paragraphs into a global paragraph list.
    Returns (paragraphs, paper_id_to_indices) where each paragraph dict is
    {"paper_id", "paper_title", "text"}.
    """
    paragraphs: list[dict] = []
    paper_to_indices: dict[str, list[int]] = {}

    for paper in papers:
        pid = paper["id"]
        title = paper["title"]
        indices = []
        full_text = paper["full_text"]
        for section_name, section_paras in zip(full_text["section_name"], full_text["paragraphs"]):
            for para in section_paras:
                para = para.strip()
                if len(para) < 20:
                    continue
                indices.append(len(paragraphs))
                paragraphs.append({
                    "paper_id": pid,
                    "paper_title": title,
                    "section": section_name or "",
                    "text": para,
                })
        paper_to_indices[pid] = indices

    return paragraphs, paper_to_indices


def _collect_questions(papers) -> list[dict]:
    """
    Extracts (question, paper_id, evidence_paragraph_texts) for every
    answerable question. Unanswerable questions are skipped -- there is no
    evidence paragraph to score retrieval against.
    """
    questions = []
    for paper in papers:
        pid = paper["id"]
        qas = paper["qas"]
        for q_text, answer_group in zip(qas["question"], qas["answers"]):
            for ans in answer_group["answer"]:
                if ans.get("unanswerable"):
                    continue
                evidence = [e.strip() for e in ans.get("evidence", []) if e and not e.startswith("FLOAT SELECTED")]
                if not evidence:
                    continue
                questions.append({
                    "question": q_text,
                    "paper_id": pid,
                    "evidence": evidence,
                })
                break  # one answer per question is enough ground truth
    return questions


def _hit_and_mrr(ranked_ids: list, gold_id, k_values=(1, 5)) -> dict:
    out = {}
    if gold_id in ranked_ids:
        rank = ranked_ids.index(gold_id) + 1
        out["mrr"] = 1.0 / rank
    else:
        rank = None
        out["mrr"] = 0.0
    for k in k_values:
        out[f"hit@{k}"] = 1.0 if (rank is not None and rank <= k) else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description="External QASPER retrieval-generalization evaluation")
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--num-papers", type=int, default=40,
                         help="Number of QASPER papers to evaluate against (max 281 for validation). Default is a bounded pilot sample.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()

    papers = _load_qasper_papers(args.split, args.num_papers)
    paragraphs, paper_to_indices = _build_paragraph_corpus(papers)
    questions = _collect_questions(papers)

    print(f"Built corpus: {len(papers)} papers, {len(paragraphs)} paragraphs, {len(questions)} answerable questions.")
    if not questions:
        print("No answerable questions in this sample -- try a larger --num-papers.")
        sys.exit(1)

    # ── Dense index (Anthology's real SPECTER2 embedder) ──────────────────
    print("Embedding paragraphs with SPECTER2 (this is the slow step)...")
    t0 = time.time()
    para_texts = [p["text"] for p in paragraphs]
    para_embeddings = embed_texts(para_texts)
    print(f"  embedded {len(para_texts)} paragraphs in {time.time()-t0:.1f}s")

    # ── Sparse index (BM25) ────────────────────────────────────────────────
    tokenized_corpus = [t.lower().split() for t in para_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    results = {"dense": [], "sparse": [], "hybrid_rrf": []}
    chunk_results = {"dense": [], "sparse": [], "hybrid_rrf": []}

    for qi, q in enumerate(questions):
        query = q["question"]
        gold_paper = q["paper_id"]
        gold_evidence_texts = set(q["evidence"])

        # Dense
        q_emb = embed_texts([query])[0]
        dense_scores = para_embeddings @ q_emb / (
            np.linalg.norm(para_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-8
        )
        dense_order = np.argsort(-dense_scores)

        # Sparse
        sparse_scores = bm25.get_scores(query.lower().split())
        sparse_order = np.argsort(-sparse_scores)

        # Hybrid RRF (k=60, standard constant)
        rrf_scores = np.zeros(len(paragraphs))
        for rank, idx in enumerate(dense_order):
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(sparse_order):
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
        hybrid_order = np.argsort(-rrf_scores)

        for strategy, order in (("dense", dense_order), ("sparse", sparse_order), ("hybrid_rrf", hybrid_order)):
            top_k_idx = order[: max(args.top_k, 20)]  # look a bit past top_k for paper-level ranking
            ranked_paper_ids = []
            ranked_para_texts = []
            for idx in top_k_idx:
                pid = paragraphs[idx]["paper_id"]
                if pid not in ranked_paper_ids:
                    ranked_paper_ids.append(pid)
                ranked_para_texts.append(paragraphs[idx]["text"])

            results[strategy].append(_hit_and_mrr(ranked_paper_ids, gold_paper, k_values=(1, args.top_k)))

            # Chunk/paragraph-level: did any of the top-k retrieved paragraphs
            # match one of the gold evidence paragraphs (exact text match)?
            ranked_hits = [1.0 if t in gold_evidence_texts else 0.0 for t in ranked_para_texts[: args.top_k]]
            chunk_results[strategy].append({
                f"hit@{args.top_k}": 1.0 if any(ranked_hits) else 0.0,
                "hit@1": ranked_hits[0] if ranked_hits else 0.0,
            })

        if (qi + 1) % 50 == 0:
            print(f"  ...{qi+1}/{len(questions)} questions evaluated")

    def _agg(rows, keys):
        return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}

    summary = {
        "dataset": "allenai/qasper",
        "dataset_license": "CC-BY-4.0",
        "split": args.split,
        "num_papers": len(papers),
        "num_paragraphs": len(paragraphs),
        "num_questions": len(questions),
        "top_k": args.top_k,
        "note": "Retrieval-only external generalization check. No generation/answer quality evaluated. See script docstring for full methodology and limitations.",
        "paper_level": {
            strat: _agg(rows, ["hit@1", f"hit@{args.top_k}", "mrr"])
            for strat, rows in results.items()
        },
        "paragraph_level": {
            strat: _agg(rows, ["hit@1", f"hit@{args.top_k}"])
            for strat, rows in chunk_results.items()
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    out_path = Path("indexes") / f"qasper_external_eval_{args.split}_{len(papers)}papers.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
