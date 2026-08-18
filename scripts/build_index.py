"""
Bulk-ingest all PDFs in a directory via the canonical pipeline
(api/services/ingest_service.py::ingest_single_paper).

Hardening (see docs/ANTHOLOGY_FULL_AUDIT.md, Phase 1 of the backend-completion
pass) over the original version of this script:

- Per-paper wall-clock timeout (`asyncio.wait_for`). A hung Docling parse
  (the documented cause of the historical 122-paper run dying silently at
  paper 71/122) can no longer stall the entire run forever -- it's logged as
  a distinct "timeout" failure and the run continues to the next paper.
- A fresh AsyncSession per paper instead of one session shared across the
  whole run. A DB-level failure on one paper can no longer leave the shared
  session in a bad state for every paper after it.
- A resumable, incrementally-written JSON status ledger
  (logs/ingestion_status.json) so a crash/hang leaves a readable per-paper
  record of exactly what succeeded/failed/timed-out, and a rerun can target
  only the papers that need it (--resume / --only-failed) instead of
  blindly re-ingesting the whole corpus.
- Failures are never silently swallowed: every failure path is logged with
  a distinct reason (parse/skip error vs. uncaught exception vs. timeout)
  and recorded in the ledger; nothing is caught-and-ignored.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.core.database import AsyncSessionLocal
from api.services.ingest_service import ingest_single_paper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAPERS_DIR      = "data/papers"
LEDGER_PATH     = "logs/ingestion_status.json"
PAPER_TIMEOUT_S = 1200  # 20 min/paper -- generous vs. the ~864s max observed on a real run,
                        # tight enough to catch a genuine hang rather than wait forever


def _load_ledger(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Ledger at %s is unreadable (%s) -- starting a fresh one", path, e)
        return {}


def _save_ledger(path: str, ledger: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename -- atomic on POSIX, avoids a truncated
    # ledger if the process is killed mid-write.
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2, default=str))
    tmp.replace(p)


async def _ingest_one(pdf_path: Path, timeout_s: int) -> dict:
    """Ingest one paper with its own session and a wall-clock timeout."""
    async with AsyncSessionLocal() as db:
        try:
            result = await asyncio.wait_for(
                ingest_single_paper(str(pdf_path), db),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            # The underlying thread-pool worker running Docling cannot be
            # force-killed from here (CPython has no safe way to cancel a
            # blocking native call in a worker thread), but abandoning the
            # await means the orchestrator itself is never stuck -- the run
            # continues. A handful of abandoned worker threads across 120
            # papers does not exhaust the default executor's pool.
            return {"error": f"TIMEOUT after {timeout_s}s", "_kind": "timeout"}
        except Exception as e:
            # DB-level or otherwise uncaught failure for this paper only.
            # Because this session is scoped to this paper, no rollback
            # bookkeeping is needed here -- the session (and its connection)
            # is simply closed by the `async with` block on the way out.
            logger.exception("Uncaught exception ingesting %s", pdf_path.name)
            return {"error": f"{type(e).__name__}: {e}", "_kind": "exception"}

    if "error" in result:
        result.setdefault("_kind", "parse_or_chunk_error")
    return result


async def run(
    papers_dir:   str  = PAPERS_DIR,
    limit:        int | None = None,
    resume:       bool = False,
    only_failed:  bool = False,
    timeout_s:    int  = PAPER_TIMEOUT_S,
    ledger_path:  str  = LEDGER_PATH,
) -> dict:
    pdf_files = sorted(Path(papers_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in %s", papers_dir)
        return {"total": 0, "attempted": 0, "succeeded": 0, "failed": 0}

    ledger = _load_ledger(ledger_path)

    if only_failed:
        pdf_files = [
            p for p in pdf_files
            if ledger.get(p.name, {}).get("status") != "success"
        ]
        logger.info("--only-failed: %d paper(s) not yet successfully ingested", len(pdf_files))
    elif resume:
        pdf_files = [
            p for p in pdf_files
            if ledger.get(p.name, {}).get("status") not in ("success",)
        ]
        logger.info("--resume: %d paper(s) remaining (skipping already-succeeded)", len(pdf_files))

    if limit is not None:
        pdf_files = pdf_files[:limit]

    logger.info("Ingesting %d PDF(s) from %s (timeout=%ds/paper)", len(pdf_files), papers_dir, timeout_s)

    succeeded  = 0
    failed     = 0
    failures_by_type: dict[str, int] = {}
    run_start = time.perf_counter()

    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info("[%d/%d] %s", i, len(pdf_files), pdf_path.name)
        paper_start = time.perf_counter()

        result = await _ingest_one(pdf_path, timeout_s)
        duration = time.perf_counter() - paper_start

        if "error" in result:
            kind = result.get("_kind", "unknown")
            logger.warning("[%d/%d] %s FAILED (%s): %s", i, len(pdf_files), pdf_path.name, kind, result["error"])
            failed += 1
            failures_by_type[kind] = failures_by_type.get(kind, 0) + 1
            ledger[pdf_path.name] = {
                "status":     "failed",
                "kind":       kind,
                "error":      result["error"],
                "duration_s": round(duration, 2),
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        else:
            logger.info(
                "[%d/%d] %s OK — %d chunks (%d figures, %d tables) in %.1fs",
                i, len(pdf_files), pdf_path.name,
                result.get("chunks", 0), result.get("figures", 0), result.get("tables", 0),
                duration,
            )
            succeeded += 1
            ledger[pdf_path.name] = {
                "status":     "success",
                "chunks":     result.get("chunks", 0),
                "figures":    result.get("figures", 0),
                "tables":     result.get("tables", 0),
                "duration_s": round(duration, 2),
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        # Persist after every paper -- a crash/hang leaves a readable ledger,
        # not just a log file that has to be grepped.
        _save_ledger(ledger_path, ledger)
        gc.collect()

    total_duration = time.perf_counter() - run_start
    summary = {
        "total":             len(pdf_files),
        "attempted":         len(pdf_files),
        "succeeded":         succeeded,
        "failed":            failed,
        "failures_by_type":  failures_by_type,
        "duration_s":        round(total_duration, 1),
    }
    logger.info("Bulk ingestion complete: %s", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk-ingest PDFs via the canonical pipeline.")
    parser.add_argument("--papers-dir", default=PAPERS_DIR, help="Directory containing PDFs to ingest")
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N PDFs (for canary runs)")
    parser.add_argument("--resume", action="store_true", help="Skip papers already marked 'success' in the ledger")
    parser.add_argument("--only-failed", action="store_true", help="Only (re-)ingest papers not marked 'success' in the ledger")
    parser.add_argument("--timeout", type=int, default=PAPER_TIMEOUT_S, help="Per-paper wall-clock timeout in seconds")
    parser.add_argument("--ledger", default=LEDGER_PATH, help="Path to the JSON status ledger")
    args = parser.parse_args()

    asyncio.run(run(
        papers_dir=args.papers_dir,
        limit=args.limit,
        resume=args.resume,
        only_failed=args.only_failed,
        timeout_s=args.timeout,
        ledger_path=args.ledger,
    ))
