from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["Benchmark"])

SCORES_PATH  = Path("indexes/eval_scores.json")
RESULTS_PATH = Path("indexes/pipeline_results.json")
QA_PATH      = Path("indexes/qa_dataset.json")

_job: dict = {"status": "idle", "progress": 0, "total": 0, "started_at": None, "error": None}


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


async def _run_one(question: str, top_k: int, use_hyde: bool) -> dict:
    import sys
    sys.path.insert(0, str(Path.cwd()))
    import src.retrieval.retriever as _retriever_module
    from src.generation.generator import generate_answer
    from api.core.database import AsyncSessionLocal

    t = time.time()
    async with AsyncSessionLocal() as db:
        chunks = await _retriever_module.retrieve(question, top_k=top_k, use_hyde=use_hyde, db=db)
    result = await generate_answer(question, chunks)
    return {
        "answer":    result["answer"],
        "contexts":  [c["text"] for c in chunks],
        "sources":   [c["metadata"]["source"] for c in chunks],
        "elapsed_s": round(time.time() - t, 2),
    }


def _run_eval_job(sample_size: int, use_judge: bool):
    global _job
    try:
        _job.update({"status": "running", "progress": 0, "error": None})

        import sys
        sys.path.insert(0, str(Path.cwd()))
        from src.evaluation.evaluator import evaluate_results, save_scores

        if not QA_PATH.exists():
            raise FileNotFoundError("No QA dataset at indexes/qa_dataset.json")

        with open(QA_PATH) as f:
            all_qa = json.load(f)
        sample = all_qa[:sample_size]
        _job["total"] = len(sample)

        results = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for i, qa in enumerate(sample):
            q = qa["question"]
            print(f"[{i+1}/{len(sample)}] {q[:70]}...")
            try:
                r = loop.run_until_complete(_run_one(q, top_k=5, use_hyde=True))
                results.append({
                    "question":     q,
                    "ground_truth": qa["answer"],
                    "source_chunk": qa.get("source_chunk", ""),
                    **r,
                    "config": {"hyde": True, "top_k": 5},
                })
            except Exception as e:
                print(f"  Failed: {e}")
                results.append({
                    "question":     q,
                    "ground_truth": qa["answer"],
                    "source_chunk": qa.get("source_chunk", ""),
                    "answer":       "ERROR",
                    "contexts":     [],
                    "sources":      [],
                    "elapsed_s":    0,
                    "error":        str(e),
                })
            _job["progress"] = i + 1

        loop.close()

        RESULTS_PATH.parent.mkdir(exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        label = f"eval_{int(time.time())}"
        scores = evaluate_results(results, label=label, run_judge=use_judge)
        save_scores(scores)
        _job["status"] = "done"

    except Exception as e:
        import traceback; traceback.print_exc()
        _job["status"] = "error"
        _job["error"] = str(e)


@router.get("/benchmark/scores")
async def get_scores():
    scores = _load_scores()
    results = _load_results()
    quick_retrieval = None
    if results:
        try:
            from src.evaluation.evaluator import compute_retrieval_metrics
            quick_retrieval = compute_retrieval_metrics(results)
        except Exception:
            pass
    return {
        "runs": scores,
        "quick_retrieval": quick_retrieval,
        "qa_count": _qa_count(),
        "result_count": len(results),
    }


class EvalRequest(BaseModel):
    sample_size: int = 20
    use_judge: bool = False


@router.post("/benchmark/run")
async def run_eval(req: EvalRequest, background_tasks: BackgroundTasks):
    if _job["status"] == "running":
        raise HTTPException(status_code=409, detail="Eval already running")
    if not QA_PATH.exists():
        raise HTTPException(status_code=400, detail="No QA dataset found")
    _job.update({"status": "starting", "started_at": time.time(), "progress": 0, "total": req.sample_size})
    background_tasks.add_task(_run_eval_job, req.sample_size, req.use_judge)
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
