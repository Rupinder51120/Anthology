"""
run_benchmark.py

Compares 5 retrieval configs end-to-end:
  1. BM25 only (baseline)
  2. FAISS only
  3. Hybrid (FAISS + BM25 + RRF), no rerank
  4. Hybrid + rerank
  5. Hybrid + rerank + HyDE
"""

import argparse
import json
from pathlib import Path

from src.benchmarker import build_qa_dataset
from src.pipeline_runner import run_pipeline_on_dataset
from src.evaluator import compare_configs

# ── save the REAL retrieve once, before any patching ──────────
import src.retriever as _ret_module
_ORIGINAL_RETRIEVE = _ret_module.retrieve   # captured before any monkey-patch


# ─── retriever patching ───────────────────────────────────────

def patch_retriever_mode(mode: str):
    import src.retriever as ret

    # Fix #3 (from previous session): always reset USE_RERANKER so earlier
    # configs don't bleed into later ones within the same process.
    ret.USE_RERANKER = False

    if mode == "bm25_only":
        def retrieve_bm25(query, top_k=5, **_):
            from src.retriever import bm25_search, load_chunks, boost_by_section_priority
            chunks = load_chunks()
            ids    = bm25_search(query, chunks, top_k=top_k)
            result = [chunks[i] for i in ids[:top_k]]
            # Fix #2: stamp rerank_score=None so format_citations never KeyErrors
            for c in result:
                c["metadata"]["rerank_score"] = None
            return result
        ret.retrieve = retrieve_bm25

    elif mode == "faiss_only":
        # Fix #1 (from previous session): _embed_query doesn't exist → use embed_texts
        def retrieve_faiss(query, top_k=5, **_):
            from src.retriever import faiss_search, load_chunks, boost_by_section_priority
            from src.embedder import embed_texts
            chunks = load_chunks()
            emb    = embed_texts([query])[0]
            ids    = faiss_search(emb, top_k=top_k)
            result = [chunks[i] for i in ids[:top_k] if i < len(chunks)]
            # Fix #2: stamp rerank_score=None
            for c in result:
                c["metadata"]["rerank_score"] = None
            return boost_by_section_priority(result)[:top_k]
        ret.retrieve = retrieve_faiss

    elif mode == "hybrid_no_rerank":
        # Fix #1 (from previous session): _embed_query doesn't exist → use embed_texts
        def retrieve_hybrid(query, top_k=5, **_):
            from src.retriever import (faiss_search, bm25_search,
                                       reciprocal_rank_fusion,
                                       deduplicate_candidates,
                                       boost_by_section_priority,
                                       load_chunks)
            from src.embedder import embed_texts
            chunks = load_chunks()
            emb    = embed_texts([query])[0]
            f_ids  = faiss_search(emb, top_k=25)
            b_ids  = bm25_search(query, chunks, top_k=25)
            fused  = reciprocal_rank_fusion(f_ids, b_ids)[:20]
            cands  = [chunks[i] for i in fused if i < len(chunks)]
            cands  = deduplicate_candidates(cands)
            # Fix #2: stamp rerank_score=None
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


# ─── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--build-qa", action="store_true")
    # Fix #9: restore --quick flag properly with two distinct qa paths
    parser.add_argument("--quick", action="store_true",
                        help="Run on qa_quick.json (15 q) for fast validation; "
                             "omit to use qa_dataset.json (50 q)")
    # Fix #1: wipe stale result + checkpoint files before a fresh benchmark run
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing result/checkpoint files before running")
    args = parser.parse_args()

    run_judge = not args.no_judge
    qa_path   = "indexes/qa_quick.json" if args.quick else "indexes/qa_dataset.json"

    # ── Fix #1: clean stale outputs so evaluator never reads old broken files ──
    if args.clean:
        import glob
        patterns = [
            "indexes/results_*.json",
            "indexes/results_*_checkpoint.json",
        ]
        removed = []
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
        #  label                    mode                use_hyde
        #("BM25 baseline",          "bm25_only",          False),
       # ("FAISS only",             "faiss_only",          False),
       # ("Hybrid no rerank",       "hybrid_no_rerank",    False),
        ("Hybrid + rerank",        "hybrid_rerank",       False),
        ("Hybrid + rerank + HyDE", "full",                True),
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

    # ── step 3: evaluate ──
    print("\n\n" + "="*55)
    print("EVALUATION")
    print("="*55)

    all_scores = compare_configs(results_map, run_judge=run_judge)

    # ── step 4: save summary ──
    summary_path = "indexes/benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    print(f"\nBenchmark complete. Summary → {summary_path}")
    if run_judge:
        print("Use the 'mean_score' column from the judge table as your resume number.")


if __name__ == "__main__":
    main()