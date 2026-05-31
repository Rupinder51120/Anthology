"""
run_benchmark.py

Compares 5 retrieval configs end-to-end:
  1. BM25 only (baseline)
  2. FAISS only
  3. Hybrid (FAISS + BM25 + RRF), no rerank
  4. Hybrid + rerank
  5. Hybrid + rerank + HyDE

Usage:
  python run_benchmark.py                  # full run with judge
  python run_benchmark.py --no-judge       # retrieval metrics only (faster, fewer tokens)
  python run_benchmark.py --build-qa       # force rebuild QA dataset first

Changes from original:
  - compare_configs now accepts run_judge flag
  - evaluate_results replaces run_ragas_eval (RAGAS removed)
  - Results paths use consistent naming
"""

import argparse
import json
from pathlib import Path

from src.benchmarker import build_qa_dataset
from src.pipeline_runner import run_pipeline_on_dataset
from src.evaluator import compare_configs, save_scores


# ─── retriever patching ───────────────────────────────────────

def patch_retriever_mode(mode: str):
    """Monkey-patches src.retriever.retrieve for the given config."""
    import src.retriever as ret

    if mode == "bm25_only":
        def retrieve_bm25(query, top_k=5, **_):
            from src.retriever import bm25_search, load_chunks
            chunks = load_chunks()
            ids = bm25_search(query, chunks, top_k=top_k)
            return [chunks[i] for i in ids[:top_k]]
        ret.retrieve = retrieve_bm25

    elif mode == "faiss_only":
        def retrieve_faiss(query, top_k=5, **_):
            from src.retriever import faiss_search, load_chunks
            from src.embedder import embed_texts
            chunks = load_chunks()
            emb    = embed_texts([query])[0]
            ids    = faiss_search(emb, top_k=top_k)
            return [chunks[i] for i in ids[:top_k] if i < len(chunks)]
        ret.retrieve = retrieve_faiss

    elif mode == "hybrid_no_rerank":
        def retrieve_hybrid(query, top_k=5, **_):
            from src.retriever import faiss_search, bm25_search, reciprocal_rank_fusion, load_chunks
            from src.embedder import embed_texts
            chunks = load_chunks()
            emb    = embed_texts([query])[0]
            f_ids  = faiss_search(emb, top_k=20)
            b_ids  = bm25_search(query, chunks, top_k=20)
            fused  = reciprocal_rank_fusion(f_ids, b_ids)[:top_k]
            return [chunks[i] for i in fused if i < len(chunks)]
        ret.retrieve = retrieve_hybrid

    elif mode == "hybrid_rerank":
        import src.retriever
        src.retriever.USE_RERANKER = True
        ret.retrieve = src.retriever.retrieve

    elif mode == "full":
        import src.retriever
        src.retriever.USE_RERANKER = True
        ret.retrieve = src.retriever.retrieve


def _safe_label(label: str) -> str:
    return label.replace(" ", "_").replace("+", "plus").replace("/", "_")


# ─── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge",  action="store_true",
                        help="Skip Groq judge eval (retrieval metrics only)")
    parser.add_argument("--build-qa",  action="store_true",
                        help="Force rebuild QA dataset even if it exists")
    args = parser.parse_args()

    run_judge = not args.no_judge
    qa_path   = "indexes/qa_dataset.json"

    # ── step 1: QA dataset ──
    if args.build_qa or not Path(qa_path).exists():
        print("Building QA dataset...")
        build_qa_dataset(output_path=qa_path, target_count=50)
    else:
        with open(qa_path) as f:
            n = len(json.load(f))
        print(f"QA dataset exists: {n} questions ({qa_path})")

    # ── step 2: run each config ──
    configs = [
        #  label                   retriever_mode    use_hyde
        ("BM25 baseline",          "bm25_only",       False),
        ("FAISS only",             "faiss_only",      False),
        ("Hybrid no rerank",       "hybrid_no_rerank",False),
        ("Hybrid + rerank",        "hybrid_rerank",   False),
        ("Hybrid + rerank + HyDE", "full",            True),
    ]

    results_map = {}

    for label, mode, use_hyde in configs:
        out_path = f"indexes/results_{_safe_label(label)}.json"
        print(f"\n{'='*55}")
        print(f"Config: {label}")
        print(f"{'='*55}")

        patch_retriever_mode(mode)

        run_pipeline_on_dataset(
            qa_path=qa_path,
            output_path=out_path,
            use_hyde=use_hyde,
            top_k=5,
            sleep_between=1.0,
        )
        results_map[label] = out_path

    # ── step 3: evaluate all configs ──
    print("\n\n" + "="*55)
    print("EVALUATION")
    print("="*55)

    all_scores = compare_configs(results_map, run_judge=run_judge)

    # ── step 4: save summary ──
    summary_path = "indexes/benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\nBenchmark complete.")
    print(f"Summary → {summary_path}")
    if run_judge:
        print("Use the 'mean_score' column from the judge table as your resume number.")
    else:
        print("Re-run without --no-judge for full judge metrics.")


if __name__ == "__main__":
    main()