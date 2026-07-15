
import json
from pathlib import Path

RESULTS_PATH = "indexes/strategy_comparison_results.json"

def analyze():
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    strategies = ["dense", "sparse", "hybrid_rrf", "hybrid_rerank"]
    # We use the first strategy's result set as the master list of questions
    first_strat = strategies[0]
    questions = data[first_strat]

    print(f"{'Query':<50} | {'Dens':<6} | {'Sprs':<6} | {'RRF':<6} | {'Rrnk':<6} | {'Analysis'}")
    print("-" * 110)

    for i, q_data in enumerate(questions):
        query_text = q_data["question"]
        gt_id = q_data.get("source_chunk_id")

        if not gt_id:
            continue

        ranks = {}
        for strat in strategies:
            results = data.get(strat, [])
            # Find the matching query in this strategy's results
            match = next((r for r in results if r["question"] == query_text), None)
            if match:
                chunk_ids = match.get("chunk_ids", [])
                try:
                    ranks[strat] = chunk_ids.index(gt_id) + 1
                except ValueError:
                    ranks[strat] = "MISS"
            else:
                ranks[strat] = "N/A"

        if ranks.get("hybrid_rerank") != 1:
            # Determine failure cause
            dense = ranks.get("dense")
            sparse = ranks.get("sparse")
            rrf = ranks.get("hybrid_rrf")
            rnk = ranks.get("hybrid_rerank")

            cause = "Unknown"
            if dense == "MISS" and sparse == "MISS":
                cause = "Increase candidate pool (Never retrieved)"
            elif rrf == "MISS" and (dense != "MISS" or sparse != "MISS"):
                cause = "Increase candidate pool (Lost during fusion)"
            elif rrf != "MISS" and rnk == "MISS":
                cause = "Increase candidate pool (Lost before rerank)"
            elif rrf != "MISS" and rnk != "MISS" and rnk > rrf:
                cause = "No retrieval change (Reranker penalty)"
            elif rrf != "MISS" and rrf > 5:
                cause = "Increase section weights (RRF rank too low)"
            elif dense != "MISS" and sparse == "MISS" and rrf > dense:
                cause = "Changing modality boosts (Sparse noise)"
            else:
                cause = "Benchmark limitation / No change"

            short_q = (query_text[:47] + '..') if len(query_text) > 47 else query_text
            print(f"{short_q:<50} | {str(dense):<6} | {str(sparse):<6} | {str(rrf):<6} | {str(rnk):<6} | {cause}")

if __name__ == '__main__':
    analyze()
