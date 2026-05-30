import json
import time
from pathlib import Path
from src.retriever import retrieve
from src.hyde import expand_query_with_hyde
from src.generator import generate_answer


def run_pipeline_on_dataset(
    qa_path: str = "indexes/qa_dataset.json",
    output_path: str = "indexes/pipeline_results.json",
    use_hyde: bool = True,
    top_k: int = 5
) -> list[dict]:

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    print(f"Running pipeline on {len(qa_pairs)} questions...")
    print(f"Config: HyDE={use_hyde}, top_k={top_k}\n")

    results = []

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        ground_truth = qa["answer"]

        print(f"[{i+1}/{len(qa_pairs)}] {question[:60]}...")

        try:
            search_query = expand_query_with_hyde(question) if use_hyde else question
            chunks = retrieve(search_query, top_k=top_k)
            result = generate_answer(question, chunks)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": result["answer"],
                "contexts": [c["text"] for c in chunks],
                "sources": [c["metadata"]["source"] for c in chunks],
                "config": {
                    "hyde": use_hyde,
                    "top_k": top_k
                }
            })

        except Exception as e:
            print(f"  Pipeline failed: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": "ERROR",
                "contexts": [],
                "sources": [],
                "config": {"hyde": use_hyde, "top_k": top_k}
            })

        time.sleep(0.3)

    Path("indexes").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {output_path}")
    return results


if __name__ == "__main__":
    run_pipeline_on_dataset()