"""
Full benchmark run. Compares:
  1. BM25 only (baseline)
  2. FAISS only
  3. Hybrid (FAISS + BM25 + RRF)
  4. Hybrid + rerank
  5. Hybrid + rerank + HyDE  ← your best config

Run: python run_benchmark.py
"""
import json
from pathlib import Path
from src.benchmarker import build_qa_dataset
from src.pipeline_runner import run_pipeline_on_dataset
from src.evaluator import compare_configs, save_scores


def patch_retriever_mode(mode: str):
    """Temporarily patch retriever to test different configs."""
    import src.retriever as ret

    if mode == "bm25_only":
        def retrieve_bm25(query, top_k=5):
            from src.retriever import bm25_search, load_chunks
            from src.embedder import embed_texts
            chunks = load_chunks()
            ids = bm25_search(query, chunks, top_k=top_k)
            return [chunks[i] for i in ids[:top_k]]
        ret.retrieve = retrieve_bm25

    elif mode == "faiss_only":
        def retrieve_faiss(query, top_k=5):
            from src.retriever import faiss_search, load_chunks
            from src.embedder import embed_texts
            chunks = load_chunks()
            emb = embed_texts([query])[0]
            ids = faiss_search(emb, top_k=top_k)
            return [chunks[i] for i in ids[:top_k] if i < len(chunks)]
        ret.retrieve = retrieve_faiss

    elif mode == "hybrid_no_rerank":
        def retrieve_hybrid(query, top_k=5):
            from src.retriever import faiss_search, bm25_search, reciprocal_rank_fusion, load_chunks
            from src.embedder import embed_texts
            chunks = load_chunks()
            emb = embed_texts([query])[0]
            f_ids = faiss_search(emb, top_k=20)
            b_ids = bm25_search(query, chunks, top_k=20)
            fused = reciprocal_rank_fusion(f_ids, b_ids)[:top_k]
            return [chunks[i] for i in fused if i < len(chunks)]
        ret.retrieve = retrieve_hybrid

    elif mode == "full":
        # restore original
        import importlib
        import src.retriever
        importlib.reload(src.retriever)
        from src.retriever import retrieve as orig
        ret.retrieve = orig


def main():
    qa_path = "indexes/qa_dataset.json"

    # Step 1: build QA dataset if not exists
    if not Path(qa_path).exists():
        print("Building QA dataset...")
        build_qa_dataset(output_path=qa_path, target_count=50)
    else:
        print(f"QA dataset exists ({qa_path}), skipping generation.")

    results_map = {}

    configs = [
        ("BM25 baseline",        "bm25_only",          False),
        ("FAISS only",           "faiss_only",          False),
        ("Hybrid no rerank",     "hybrid_no_rerank",    False),
        ("Hybrid + rerank",      "full",                False),
        ("Hybrid + rerank+HyDE", "full",                True),
    ]

    for label, mode, use_hyde in configs:
        out_path = f"indexes/results_{label.replace(' ', '_').replace('+', 'plus')}.json"
        print(f"\n{'='*50}")
        print(f"Running config: {label}")
        print(f"{'='*50}")

        patch_retriever_mode(mode)

        run_pipeline_on_dataset(
            qa_path=qa_path,
            output_path=out_path,
            use_hyde=use_hyde,
            top_k=5
        )
        results_map[label] = out_path

    # Step 2: compare all
    print("\n\nFINAL COMPARISON")
    all_scores = compare_configs(results_map)

    # Step 3: save
    with open("indexes/benchmark_summary.json", "w") as f:
        json.dump(all_scores, f, indent=2)

    print("\nFull benchmark complete. Check indexes/benchmark_summary.json")
    print("Use these numbers on your resume.")


if __name__ == "__main__":
    main()