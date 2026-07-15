from __future__ import annotations
import json
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from src.evaluation.pipeline_runner import run_pipeline_on_dataset_async

router = APIRouter(prefix="/api/v1", tags=["Benchmark"])

SCORES_PATH  = Path("indexes/eval_scores.json")
RESULTS_PATH = Path("indexes/pipeline_results.json")
QA_PATH      = Path("indexes/qa_dataset.json")

# GATE 3: comparison-run artifacts, separate from the single-strategy job above
COMPARISON_RESULTS_PATH = Path("indexes/strategy_comparison_results.json")
COMPARISON_SCORES_PATH  = Path("indexes/strategy_comparison_scores.json")

STRATEGIES = ["sparse", "dense", "dense_hyde", "hybrid_rrf", "hybrid_rerank", "dense_rerank"]
# strategies that call HyDE internally (used for benchmark bookkeeping/labels only)
HYDE_STRATEGIES = {"dense_hyde"}

_job: dict = {"status": "idle", "progress": 0, "total": 0, "started_at": None, "error": None}
_comparison_job: dict = {"status": "idle", "progress": 0, "total": 0, "started_at": None, "error": None, "current_strategy": None}


def _load_scores() -> dict:
    if SCORES_PATH.exists():
        with open(SCORES_PATH) as f:
            return json.load(f)
    return {}

def _load_results() -> list[dict]:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []

def _qa_count() -> int:
    if QA_PATH.exists():
        with open(QA_PATH) as f:
            return len(json.load(f))
    return 0


async def _run_one(question: str, top_k: int, use_hyde: bool, strategy: str = "hybrid_rerank") -> dict:
    import sys
    sys.path.insert(0, str(Path.cwd()))
    import src.retrieval.retriever as _retriever_module
    from src.generation.generator import generate_answer
    from api.core.database import AsyncSessionLocal

    t = time.time()
    async with AsyncSessionLocal() as db:
        chunks = await _retriever_module.retrieve(
            question, top_k=top_k, use_hyde=use_hyde, db=db, strategy=strategy
        )
    result = await generate_answer(question, chunks)
    return {
        "answer":    result["answer"],
        "contexts":  [c["text"] for c in chunks],
        "sources":   [c["metadata"]["source"] for c in chunks],
        "chunk_ids": [c["metadata"]["chunk_id"] for c in chunks],
        "elapsed_s": round(time.time() - t, 2),
    }


async def _run_eval_job(req: EvalRequest):
    """
    Native async background task — runs on uvicorn's existing event loop.
    Converged to use the canonical pipeline runner.
    """
    global _job
    try:
        _job.update({"status": "running", "progress": 0, "error": None})
        _job["total"] = req.sample_size

        # Use canonical orchestrator
        results = await run_pipeline_on_dataset_async(
            qa_path=str(QA_PATH),
            output_path=str(RESULTS_PATH),
            sample_size=req.sample_size,
            use_hyde=False,  # HyDE is enabled only by the dense_hyde strategy
            top_k=req.top_k,
            strategy=req.strategy,
            run_judge=req.use_judge,
        )

        # The pipeline runner now handles saving results and computing/saving metrics.
        # We just need to update the job status.
        _job["status"] = "done"

    except Exception as e:
        import traceback; traceback.print_exc()
        _job["status"] = "error"
        _job["error"] = str(e)


async def _run_comparison_job(sample_size: int, strategies: list[str]):
    """
    GATE 3: runs the same QA sample through every requested strategy and
    scores each independently (paper-level + chunk-level Hit@5/MRR/nDCG,
    plus mean latency), so strategies can be compared side by side rather
    than only ever benchmarking the bundled production pipeline.
    """
    global _comparison_job
    try:
        _comparison_job.update({"status": "running", "progress": 0, "error": None})

        import sys
        sys.path.insert(0, str(Path.cwd()))
        from src.evaluation.evaluator import (
            compute_retrieval_metrics,
            compute_chunk_retrieval_metrics,
        )

        if not QA_PATH.exists():
            raise FileNotFoundError("No QA dataset at indexes/qa_dataset.json")

        with open(QA_PATH) as f:
            all_qa = json.load(f)
        sample = all_qa[:sample_size]
        total_steps = len(sample) * len(strategies)
        _comparison_job["total"] = total_steps

        all_results: dict[str, list[dict]] = {s: [] for s in strategies}
        step = 0

        for strategy in strategies:
            _comparison_job["current_strategy"] = strategy
            use_hyde = strategy in HYDE_STRATEGIES
            for i, qa in enumerate(sample):
                q = qa["question"]
                print(f"[{strategy}] [{i+1}/{len(sample)}] {q[:60]}...")
                try:
                    r = await _run_one(q, top_k=5, use_hyde=use_hyde, strategy=strategy)
                    all_results[strategy].append({
                        "question":        q,
                        "ground_truth":    qa["answer"],
                        "source_chunk":    qa.get("source_chunk", ""),
                        "source_paper":    qa.get("source_paper", qa.get("source_chunk", "")),
                        "source_chunk_id": qa.get("source_chunk_id", ""),
                        "content_type":    qa.get("content_type", "text"),
                        **r,
                        "config": {"strategy": strategy, "hyde": use_hyde, "top_k": 5},
                    })
                except Exception as e:
                    print(f"  Failed: {e}")
                    all_results[strategy].append({
                        "question":        q,
                        "ground_truth":    qa["answer"],
                        "source_chunk_id": qa.get("source_chunk_id", ""),
                        "content_type":    qa.get("content_type", "text"),
                        "answer":          "ERROR",
                        "contexts":        [], "sources": [], "chunk_ids": [],
                        "elapsed_s":       0,
                        "error":           str(e),
                        "config":          {"strategy": strategy, "hyde": use_hyde, "top_k": 5},
                    })
                step += 1
                _comparison_job["progress"] = step

        COMPARISON_RESULTS_PATH.parent.mkdir(exist_ok=True)
        with open(COMPARISON_RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

        # score each strategy independently — paper-level, chunk-level, latency
        comparison_scores = {}
        for strategy, results in all_results.items():
            paper_metrics = compute_retrieval_metrics(results)
            chunk_metrics = compute_chunk_retrieval_metrics(results)
            valid = [r for r in results if r.get("elapsed_s", 0) > 0]
            mean_latency = round(sum(r["elapsed_s"] for r in valid) / len(valid), 3) if valid else None
            error_count = sum(1 for r in results if r.get("answer") == "ERROR")

            # modality slice: Hit@5 broken out by content_type, per your earlier request
            by_modality: dict[str, list[dict]] = {}
            for r in results:
                by_modality.setdefault(r.get("content_type", "text"), []).append(r)
            modality_breakdown = {
                ct: compute_chunk_retrieval_metrics(rs) for ct, rs in by_modality.items()
            }

            comparison_scores[strategy] = {
                "paper_level":        paper_metrics,
                "chunk_level":        chunk_metrics,
                "by_modality":        modality_breakdown,
                "mean_latency_s":     mean_latency,
                "error_count":        error_count,
                "sample_size":        len(results),
            }

        with open(COMPARISON_SCORES_PATH, "w") as f:
            json.dump(comparison_scores, f, indent=2)

        _comparison_job["status"] = "done"
        _comparison_job["current_strategy"] = None

    except Exception as e:
        import traceback; traceback.print_exc()
        _comparison_job["status"] = "error"
        _comparison_job["error"] = str(e)


@router.get("/benchmark/scores")
async def get_scores():
    scores = _load_scores()
    results = _load_results()
    quick_retrieval = None
    quick_chunk_retrieval = None
    if results:
        try:
            from src.evaluation.evaluator import compute_retrieval_metrics, compute_chunk_retrieval_metrics
            quick_retrieval = compute_retrieval_metrics(results)
            quick_chunk_retrieval = compute_chunk_retrieval_metrics(results)
        except Exception:
            pass
    return {
        "runs": scores,
        "quick_retrieval": quick_retrieval,
        "quick_chunk_retrieval": quick_chunk_retrieval,
        "qa_count": _qa_count(),
        "result_count": len(results),
    }


class EvalRequest(BaseModel):
    sample_size: int = 20
    use_judge: bool = False
    strategy: str = "hybrid_rerank"
    top_k: int = 5


@router.post("/benchmark/run")
async def run_eval(req: EvalRequest, background_tasks: BackgroundTasks):
    if _job["status"] == "running":
        raise HTTPException(status_code=409, detail="Eval already running")
    if not QA_PATH.exists():
        raise HTTPException(status_code=400, detail="No QA dataset found")
    _job.update({"status": "starting", "started_at": time.time(), "progress": 0, "total": req.sample_size})
    background_tasks.add_task(_run_eval_job, req)
    return {"status": "started", "sample_size": req.sample_size}


@router.get("/benchmark/status")
async def eval_status():
    return {
        **_job,
        "elapsed_s": round(time.time() - _job["started_at"], 1) if _job.get("started_at") else None,
    }


@router.get("/benchmark/results")
async def get_results(limit: int = 50):
    results = _load_results()
    return {"total": len(results), "results": results[:limit]}


# ── GATE 3: strategy comparison endpoints ───────────────────────────────

class ComparisonRequest(BaseModel):
    sample_size: int = 20
    strategies: list[str] = STRATEGIES


@router.post("/benchmark/compare")
async def run_comparison(req: ComparisonRequest, background_tasks: BackgroundTasks):
    if _comparison_job["status"] == "running":
        raise HTTPException(status_code=409, detail="Comparison already running")
    if not QA_PATH.exists():
        raise HTTPException(status_code=400, detail="No QA dataset found")
    invalid = set(req.strategies) - set(STRATEGIES)
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown strategies: {sorted(invalid)}")
    _comparison_job.update({
        "status": "starting", "started_at": time.time(), "progress": 0,
        "total": req.sample_size * len(req.strategies), "current_strategy": None,
    })
    background_tasks.add_task(_run_comparison_job, req.sample_size, req.strategies)
    return {"status": "started", "sample_size": req.sample_size, "strategies": req.strategies}


@router.get("/benchmark/compare/status")
async def comparison_status():
    return {
        **_comparison_job,
        "elapsed_s": round(time.time() - _comparison_job["started_at"], 1) if _comparison_job.get("started_at") else None,
    }


@router.get("/benchmark/compare/results")
async def comparison_results():
    """
    Returns the per-strategy score table: paper-level + chunk-level
    Hit@5/MRR/nDCG@5, modality breakdown, mean latency, and error count —
    for every strategy in the last comparison run.
    """
    if not COMPARISON_SCORES_PATH.exists():
        raise HTTPException(status_code=404, detail="No comparison run found yet")
    with open(COMPARISON_SCORES_PATH) as f:
        return json.load(f)