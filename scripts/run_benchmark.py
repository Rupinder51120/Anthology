"""
run_benchmark.py

Compares retrieval configs end-to-end.
Now includes QASPER-style unbiased evaluation.

Modes:
  python run_benchmark.py                  # run existing biased benchmark
  python run_benchmark.py --build-qa       # regenerate biased QA dataset
  python run_benchmark.py --build-qasper   # generate QASPER-style benchmark (uses local Ollama)
  python run_benchmark.py --qasper         # run cross-benchmark comparison
  python run_benchmark.py --clean          # wipe stale result files first
  python run_benchmark.py --quick          # use 15-question fast dataset
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.evaluation.benchmarker import build_qa_dataset
from src.evaluation.pipeline_runner import run_pipeline_on_dataset
from src.evaluation.evaluator import compare_configs

# ── save the REAL retrieve once, before any patching ──────────
import src.retrieval.retriever as _ret_module
_ORIGINAL_RETRIEVE = _ret_module.retrieve


# ─────────────────────────────────────────────────────────────
# RETRIEVER PATCHING  (unchanged from your original)
# ─────────────────────────────────────────────────────────────

def patch_retriever_mode(mode: str):
    import src.retrieval.retriever as ret
    ret.USE_RERANKER = False

    if mode == "bm25_only":
        def retrieve_bm25(query, top_k=5, **_):
            from src.retrieval.retriever import bm25_search, load_chunks, boost_by_section_priority
            chunks = load_chunks()
            ids    = bm25_search(query, chunks, top_k=top_k)
            result = [chunks[i] for i in ids[:top_k]]
            for c in result:
                c["metadata"]["rerank_score"] = None
            return result
        ret.retrieve = retrieve_bm25

    elif mode == "faiss_only":
        def retrieve_faiss(query, top_k=5, **_):
            from src.retrieval.retriever import faiss_search, load_chunks, boost_by_section_priority
            from src.retrieval.embedder import embed_texts
            chunks = load_chunks()
            emb    = embed_texts([query])[0]
            ids    = faiss_search(emb, top_k=top_k)
            result = [chunks[i] for i in ids[:top_k] if i < len(chunks)]
            for c in result:
                c["metadata"]["rerank_score"] = None
            return boost_by_section_priority(result)[:top_k]
        ret.retrieve = retrieve_faiss

    elif mode == "hybrid_no_rerank":
        def retrieve_hybrid(query, top_k=5, **_):
            from src.retrieval.retriever import (faiss_search, bm25_search,
                                       reciprocal_rank_fusion,
                                       deduplicate_candidates,
                                       boost_by_section_priority,
                                       load_chunks)
            from src.retrieval.embedder import embed_texts
            chunks = load_chunks()
            emb    = embed_texts([query])[0]
            f_ids  = faiss_search(emb, top_k=25)
            b_ids  = bm25_search(query, chunks, top_k=25)
            fused  = reciprocal_rank_fusion(f_ids, b_ids)[:20]
            cands  = [chunks[i] for i in fused if i < len(chunks)]
            cands  = deduplicate_candidates(cands)
            for c in cands:
                c["metadata"]["rerank_score"] = None
            return boost_by_section_priority(cands)[:top_k]
        ret.retrieve = retrieve_hybrid

    elif mode == "hybrid_rerank":
        ret.USE_RERANKER = True
        def retrieve_rerank(query, top_k=5, **_):
            return _ORIGINAL_RETRIEVE(query, top_k=top_k, use_hyde=False)
        ret.retrieve = retrieve_rerank

    elif mode == "full":
        ret.USE_RERANKER = True
        def retrieve_full(query, top_k=5, **_):
            return _ORIGINAL_RETRIEVE(query, top_k=top_k, use_hyde=True)
        ret.retrieve = retrieve_full


def _safe_label(label: str) -> str:
    return label.replace(" ", "_").replace("+", "plus").replace("/", "_")


# ─────────────────────────────────────────────────────────────
# QASPER INTEGRATION
# ─────────────────────────────────────────────────────────────

def _build_paper_index(chunks_path: str = "indexes/chunks_metadata.json") -> dict:
    """
    Groups chunks by paper (source filename).
    Extracts abstract text from chunks where section == "abstract".

    Returns:
        {
          "paper.pdf": {
            "title":    "...",
            "abstract": "...",   # empty string if no abstract chunk found
            "chunks":   [{"id": "paper.pdf::42", "text": "..."}]
          }
        }
    """
    with open(chunks_path) as f:
        chunks = json.load(f)

    papers = defaultdict(lambda: {"title": "", "abstract": "", "chunks": []})

    for i, chunk in enumerate(chunks):
        meta   = chunk["metadata"]
        source = meta["source"]
        papers[source]["title"] = meta.get("title", source)

        chunk_id = f"{source}::{i}"

        if meta.get("section", "").lower() == "abstract":
            papers[source]["abstract"] += " " + chunk["text"]

        papers[source]["chunks"].append({
            "id":   chunk_id,
            "text": chunk["text"],
        })

    # Strip whitespace from abstract
    for p in papers.values():
        p["abstract"] = p["abstract"].strip()

    return dict(papers)


def build_qasper_benchmark(
    output_path: str = "indexes/qa_qasper.json",
    questions_per_paper: int = 2,
    max_papers: int = 20,
):
    """
    Generates QASPER-style QA pairs using local Ollama (qwen2.5:7b).
    Questions are written from title + abstract ONLY.
    Answer evidence is located in the full chunk set afterward.
    This produces low-lexical-overlap questions that BM25 cannot game.
    """
    import requests
    import re
    import time

    OLLAMA_URL   = "http://localhost:11434/api/chat"
    OLLAMA_MODEL = "qwen2.5:7b"

    def ollama_call(prompt: str, max_tokens: int = 500) -> str:
        """Call local Ollama and return raw text response."""
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": 0.7, "num_predict": max_tokens},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def parse_json(raw: str) -> dict:
        """Strip markdown fences and parse JSON — same helper as benchmarker.py."""
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found: {raw[:200]!r}")
        return json.loads(raw[start:end])

    paper_index = _build_paper_index()

    # Filter papers that have a real abstract
    eligible = {
        src: data for src, data in paper_index.items()
        if len(data["abstract"]) > 100
    }

    print(f"Found {len(eligible)} papers with abstracts (need >100 chars)")
    print(f"Will process up to {max_papers} papers, "
          f"{questions_per_paper} questions each")

    # Limit to max_papers
    selected = list(eligible.items())[:max_papers]

    QUESTION_PROMPT = """You are a researcher who has read ONLY the title and abstract below.
Write ONE question you would ask after reading just these — something that requires
reading the full paper's methods or results to answer.

Rules:
- Must NOT be answerable from the abstract alone
- Must NOT contain phrases copied verbatim from the abstract
- Avoid generic questions like "What is this paper about?"
- Prefer: "Why did the authors choose X?", "How does Y compare to Z?",
  "What assumption does the method make about W?"

Title: {title}
Abstract: {abstract}

Respond ONLY in JSON (no markdown):
{{"question": "...", "question_type": "factoid|abstractive|comparative", "difficulty": "easy|medium|hard"}}"""

    ANSWER_PROMPT = """You are a research assistant. Given a question and paper chunks,
find the best answer using ONLY the chunk text provided.

Question: {question}

Chunks:
{chunks}

Respond ONLY in JSON (no markdown):
{{"answer": "...", "supporting_chunk_ids": ["id1", "id2"], "is_answerable": true}}

If no chunk answers the question, set is_answerable to false."""

    all_pairs = []

    for paper_idx, (source, data) in enumerate(selected):
        title    = data["title"]
        abstract = data["abstract"]
        chunks   = data["chunks"]

        print(f"\n[{paper_idx+1}/{len(selected)}] {title[:60]}")
        print(f"  Abstract: {len(abstract)} chars | Chunks: {len(chunks)}")

        generated = 0
        attempts  = 0

        while generated < questions_per_paper and attempts < questions_per_paper * 3:
            attempts += 1

            # Step 1: generate question from title + abstract only
            try:
                raw    = ollama_call(QUESTION_PROMPT.format(
                    title=title, abstract=abstract[:1500]
                ), max_tokens=300)
                q_data = parse_json(raw)
                question = q_data.get("question", "").strip()
                if not question:
                    continue
            except Exception as e:
                print(f"  Question gen failed: {e}")
                time.sleep(1)
                continue

            print(f"  Q: {question[:80]}")

            # Step 2: find answer in chunks (pass top 15 chunks to stay under token limit)
            chunk_sample = chunks[:15]
            chunks_text  = "\n\n".join(
                f"[{c['id']}]\n{c['text'][:600]}"
                for c in chunk_sample
            )

            try:
                raw    = ollama_call(ANSWER_PROMPT.format(
                    question=question,
                    chunks=chunks_text
                ), max_tokens=500)
                a_data = parse_json(raw)

                if not a_data.get("is_answerable", False):
                    print(f"  → unanswerable, skipping")
                    continue

            except Exception as e:
                print(f"  Answer gen failed: {e}")
                time.sleep(1)
                continue

            all_pairs.append({
                "question":            question,
                "answer":              a_data.get("answer", ""),
                "source_chunk":        source,         # paper filename = paper ID
                "supporting_chunks":   a_data.get("supporting_chunk_ids", []),
                "question_type":       q_data.get("question_type", "abstractive"),
                "difficulty":          q_data.get("difficulty", "medium"),
                "generation_method":   "qasper_style",
                "section":             "multi-section",
            })
            generated += 1
            print(f"  ✓ pair {generated}/{questions_per_paper} saved")
            time.sleep(0.3)

    Path("indexes").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\nQASPER benchmark saved: {len(all_pairs)} pairs → {output_path}")
    return all_pairs


# ─────────────────────────────────────────────────────────────
# RETRIEVER ADAPTER FOR EVAL
# ─────────────────────────────────────────────────────────────

def make_retriever_fn(mode: str, top_k: int = 10):
    """
    Wraps your existing patched retriever into the signature
    the metrics layer expects:
        question -> list[str]   (paper source filenames)

    We use source filename as the ID because your chunks don't
    have explicit IDs but do have metadata["source"].
    """
    patch_retriever_mode(mode)
    import src.retrieval.retriever as ret
    # Capture the patched function right now so it doesn't get
    # overwritten by the next patch_retriever_mode call.
    captured = ret.retrieve

    def retriever_fn(question: str) -> list[str]:
        results = captured(question, top_k=top_k)
        return [r["metadata"]["source"] for r in results]

    return retriever_fn


# ─────────────────────────────────────────────────────────────
# METRICS  (self-contained, no extra files needed)
# ─────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 1.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for rank, item in enumerate(retrieved, 1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def lexical_overlap(question: str, answer_text: str) -> float:
    """Jaccard overlap of content words between question and answer chunk."""
    import re as _re
    STOP = {
        "the","a","an","is","are","was","were","be","been","have","has",
        "had","do","does","did","will","would","could","should","of","in",
        "on","at","to","for","with","by","from","and","or","but","not",
        "this","that","it","what","how","why","which","used","using","paper",
    }
    def tokens(t):
        return {w for w in _re.findall(r'\b[a-z]{3,}\b', t.lower())
                if w not in STOP}
    q, a = tokens(question), tokens(answer_text)
    if not q:
        return 0.0
    union = q | a
    return len(q & a) / len(union) if union else 0.0


# ─────────────────────────────────────────────────────────────
# QASPER EVAL RUNNER
# ─────────────────────────────────────────────────────────────

def run_qasper_eval(
    biased_path:  str = "indexes/qa_dataset.json",
    qasper_path:  str = "indexes/qa_qasper.json",
    k_values: list   = None,
):
    """
    Loads both benchmarks, runs all retriever configs on each,
    prints a cross-benchmark comparison table.
    The drop in BM25 performance from biased→qasper is your key finding.
    """
    k_values = k_values or [1, 5, 10]

    # ── load benchmarks ──────────────────────────────────────
    def load_qa(path):
        with open(path) as f:
            return json.load(f)

    benchmarks = {}
    if Path(biased_path).exists():
        benchmarks["biased (chunk-derived)"] = load_qa(biased_path)
        print(f"Loaded biased benchmark:  {len(benchmarks['biased (chunk-derived)'])} pairs")
    else:
        print(f"WARNING: biased benchmark not found at {biased_path}")

    if Path(qasper_path).exists():
        benchmarks["qasper-style"] = load_qa(qasper_path)
        print(f"Loaded QASPER benchmark:  {len(benchmarks['qasper-style'])} pairs")
    else:
        print(f"WARNING: QASPER benchmark not found at {qasper_path}")
        print("Run with --build-qasper first.")
        return

    if not benchmarks:
        print("No benchmarks found. Exiting.")
        return

    # ── retriever configs to compare ─────────────────────────
    retriever_configs = [
        ("BM25",          "bm25_only"),
        ("FAISS",         "faiss_only"),
        ("Hybrid",        "hybrid_no_rerank"),
        ("Hybrid+Rerank", "hybrid_rerank"),
        ("Hybrid+HyDE",   "full"),
    ]

    # ── run eval ─────────────────────────────────────────────
    # results[benchmark_name][method_name][k] = {recall, hit, mrr}
    results = defaultdict(lambda: defaultdict(dict))

    for bm_name, qa_pairs in benchmarks.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on: {bm_name}")
        print(f"{'='*60}")

        # Compute mean lexical overlap for this benchmark
        overlaps = [
            lexical_overlap(qa["question"], qa.get("answer", ""))
            for qa in qa_pairs
        ]
        mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        high_bias    = sum(1 for o in overlaps if o > 0.35)
        print(f"  Mean lexical overlap: {mean_overlap:.4f}  "
              f"| High-bias questions: {high_bias}/{len(overlaps)}")
        verdict = (
            "HIGH BIAS — favors BM25" if mean_overlap > 0.35 else
            "MODERATE BIAS"           if mean_overlap > 0.15 else
            "LOW BIAS — realistic"
        )
        print(f"  Verdict: {verdict}\n")

        for method_name, mode in retriever_configs:
            print(f"  Running {method_name}...")
            retriever_fn = make_retriever_fn(mode, top_k=max(k_values))

            recall_sums = defaultdict(float)
            hit_sums    = defaultdict(float)
            mrr_sum     = 0.0

            for qa in qa_pairs:
                question   = qa["question"]
                gold_paper = qa["source_chunk"]   # filename is the paper ID
                gold       = {gold_paper}

                retrieved = retriever_fn(question)

                mrr_sum += mrr(retrieved, gold)
                for k in k_values:
                    recall_sums[k] += recall_at_k(retrieved, gold, k)
                    hit_sums[k]    += hit_at_k(retrieved, gold, k)

            n = len(qa_pairs)
            results[bm_name][method_name] = {
                "mrr":     mrr_sum / n,
                "recall":  {k: recall_sums[k] / n for k in k_values},
                "hit":     {k: hit_sums[k] / n    for k in k_values},
                "n":       n,
            }

    # ── print per-benchmark tables ────────────────────────────
    for bm_name, method_results in results.items():
        k = 5  # primary display K
        print(f"\n{'='*65}")
        print(f"  Results on: {bm_name}  (primary K={k})")
        print(f"{'='*65}")
        print(f"  {'Method':20s}  {'Recall@'+str(k):12s}  {'Hit@'+str(k):10s}  MRR")
        print(f"  {'-'*55}")
        for method, scores in sorted(
            method_results.items(),
            key=lambda x: -x[1]["recall"].get(k, 0)
        ):
            print(f"  {method:20s}  "
                  f"{scores['recall'].get(k, 0):.4f}        "
                  f"{scores['hit'].get(k, 0):.4f}      "
                  f"{scores['mrr']:.4f}")

    # ── cross-benchmark comparison (the key table) ────────────
    if len(results) >= 2:
        k = 5
        bm_names = list(results.keys())
        print(f"\n{'='*75}")
        print(f"  CROSS-BENCHMARK COMPARISON  Recall@{k}")
        print(f"  Drop = biased→qasper.  Large BM25 drop confirms benchmark bias.")
        print(f"{'='*75}")

        all_methods = list(list(results.values())[0].keys())
        header = f"  {'Method':20s}  "
        header += "  ".join(f"{b[:22]:22s}" for b in bm_names)
        header += "  Drop"
        print(header)
        print(f"  {'-'*70}")

        for method in all_methods:
            scores = [
                results[bm].get(method, {}).get("recall", {}).get(k, None)
                for bm in bm_names
            ]
            valid = [s for s in scores if s is not None]
            drop_str = ""
            if len(valid) >= 2:
                delta    = valid[-1] - valid[0]
                drop_str = f"{delta:+.4f}"
                if delta < -0.10:
                    drop_str += "  ← BIAS CONFIRMED"

            row = f"  {method:20s}  "
            row += "  ".join(
                f"{s:.4f}                " if s is not None else "N/A                   "
                for s in scores
            )
            row += f"  {drop_str}"
            print(row)

        print(f"{'='*75}")
        print("\n  Interpretation:")
        print("  • BM25 drop > 0.15 → your chunk-derived questions are lexically biased")
        print("  • Hybrid/Semantic drop < 0.05 → those methods generalise to real queries")
        print("  • Report BOTH benchmarks in writeup — that IS the finding")

    # ── save results ──────────────────────────────────────────
    out_path = "indexes/qasper_eval_results.json"
    # Convert defaultdict to regular dict for JSON serialisation
    serialisable = {
        bm: {
            method: {
                **scores,
                "recall": {str(k): v for k, v in scores["recall"].items()},
                "hit":    {str(k): v for k, v in scores["hit"].items()},
            }
            for method, scores in method_results.items()
        }
        for bm, method_results in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n  Full results saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge",      action="store_true")
    parser.add_argument("--build-qa",      action="store_true")
    parser.add_argument("--build-qasper",  action="store_true",
                        help="Generate QASPER-style benchmark via local Ollama (qwen2.5:7b)")
    parser.add_argument("--qasper",        action="store_true",
                        help="Run cross-benchmark comparison (biased vs QASPER)")
    parser.add_argument("--quick",         action="store_true",
                        help="Use qa_quick.json (15 q) for fast validation")
    parser.add_argument("--clean",         action="store_true",
                        help="Delete stale result/checkpoint files before running")
    parser.add_argument("--max-papers",    type=int, default=20,
                        help="Max papers for QASPER benchmark generation (default 20)")
    parser.add_argument("--questions-per-paper", type=int, default=2,
                        help="QASPER questions per paper (default 2)")
    args = parser.parse_args()

    run_judge = not args.no_judge
    qa_path   = "indexes/qa_quick.json" if args.quick else "indexes/qa_dataset.json"

    # ── clean stale files ─────────────────────────────────────
    if args.clean:
        import glob
        patterns = ["indexes/results_*.json", "indexes/results_*_checkpoint.json"]
        removed  = []
        for pat in patterns:
            for f in glob.glob(pat):
                Path(f).unlink()
                removed.append(f)
        if removed:
            print(f"Cleaned {len(removed)} stale file(s):")
            for f in removed:
                print(f"  {f}")
        else:
            print("No stale files to clean.")

    # ── QASPER benchmark generation ───────────────────────────
    if args.build_qasper:
        build_qasper_benchmark(
            output_path="indexes/qa_qasper.json",
            questions_per_paper=args.questions_per_paper,
            max_papers=args.max_papers,
        )
        return

    # ── QASPER cross-benchmark evaluation ─────────────────────
    if args.qasper:
        run_qasper_eval(
            biased_path="indexes/qa_dataset.json",
            qasper_path="indexes/qa_qasper.json",
        )
        return

    # ── original benchmark flow (unchanged) ───────────────────
    if args.build_qa or not Path(qa_path).exists():
        print("Building QA dataset...")
        build_qa_dataset(output_path=qa_path, target_count=50)
    else:
        with open(qa_path) as f:
            n = len(json.load(f))
        print(f"QA dataset exists: {n} questions ({qa_path})")

    configs = [
        ("BM25 baseline",    "bm25_only",       False),
        ("FAISS only",       "faiss_only",       False),
        ("Hybrid",           "hybrid_no_rerank", False),
        ("Hybrid + rerank",  "hybrid_rerank",    False),
        ("Hybrid + HyDE",  "full",             True),  # skip — slow on CPU
    ]

    results_map = {}
    for label, mode, use_hyde in configs:
        out_path = f"indexes/results_{_safe_label(label)}.json"
        print(f"\n{'='*55}")
        print(f"Config: {label}  |  mode={mode}  |  hyde={use_hyde}")
        print(f"{'='*55}")
        patch_retriever_mode(mode)
        run_pipeline_on_dataset(
            qa_path=qa_path,
            output_path=out_path,
            use_hyde=use_hyde,
            top_k=7,
            sleep_between=0,
        )
        results_map[label] = out_path

    print("\n\n" + "="*55)
    print("EVALUATION")
    print("="*55)
    all_scores = compare_configs(results_map, run_judge=run_judge)

    summary_path = "indexes/benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\nBenchmark complete. Summary → {summary_path}")
    if run_judge:
        print("Use the 'mean_score' column from the judge table as your resume number.")


if __name__ == "__main__":
    main()