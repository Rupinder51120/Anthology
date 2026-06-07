import json
from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter(prefix="/api/v1", tags=["Benchmark"])


@router.get("/benchmark")
async def get_benchmark():
    """Get latest benchmark results."""
    summary_path = Path("indexes/benchmark_summary.json")
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark results found. Run scripts/run_benchmark.py first."
        )
    with open(summary_path) as f:
        return json.load(f)


@router.get("/benchmark/report")
async def get_build_report():
    """Get latest index build report."""
    report_path = Path("indexes/build_report.json")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No build report found.")
    with open(report_path) as f:
        return json.load(f)
