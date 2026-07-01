
from __future__ import annotations

import asyncio
import gc
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.core.database import AsyncSessionLocal
from api.services.ingest_service import ingest_single_paper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAPERS_DIR = "data/papers"


async def run(papers_dir: str = PAPERS_DIR) -> dict:
    
    pdf_files = sorted(Path(papers_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in %s", papers_dir)
        return {"total": 0, "succeeded": 0, "failed": 0}

    logger.info("Found %d PDFs in %s", len(pdf_files), papers_dir)

    succeeded = 0
    failed    = 0

    async with AsyncSessionLocal() as db:
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info("[%d/%d] %s", i, len(pdf_files), pdf_path.name)

            try:
                result = await ingest_single_paper(str(pdf_path), db)
            except Exception as e:
              
                logger.error("[%d/%d] %s FAILED (uncaught): %s", i, len(pdf_files), pdf_path.name, e)
                failed += 1
                continue

            if "error" in result:
                logger.warning("[%d/%d] %s SKIPPED: %s", i, len(pdf_files), pdf_path.name, result["error"])
                failed += 1
            else:
                logger.info(
                    "[%d/%d] %s OK — %d chunks (%d figures, %d tables)",
                    i, len(pdf_files), pdf_path.name,
                    result.get("chunks", 0), result.get("figures", 0), result.get("tables", 0),
                )
                succeeded += 1

            
            del result
            gc.collect()

    summary = {"total": len(pdf_files), "succeeded": succeeded, "failed": failed}
    logger.info("Bulk ingestion complete: %s", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bulk-ingest all PDFs in a directory via the canonical pipeline.")
    parser.add_argument("--papers-dir", default=PAPERS_DIR, help="Directory containing PDFs to ingest")
    args = parser.parse_args()

    asyncio.run(run(args.papers_dir))