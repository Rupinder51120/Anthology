# scripts/canary_ingest.py
from pathlib import Path
import sys
import asyncio
import logging

# Add project root to path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

# Configure logging to see metadata resolution logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from api.core.database import AsyncSessionLocal
from api.services.ingest_service import ingest_single_paper

async def main():
    papers_dir = Path("data/papers")
    if not papers_dir.exists():
        logger.error("Papers directory not found: %s", papers_dir.absolute())
        return

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", papers_dir)
        return

    # Test a small subset
    subset = pdfs[:20]
    logger.info("Starting canary ingestion for %d papers...", len(subset))

    async with AsyncSessionLocal() as db:
        for i, pdf in enumerate(subset, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(subset)}] Processing: {pdf.name}")
            print(f"{'='*60}")

            try:
                result = await ingest_single_paper(str(pdf), db)
                if "error" in result:
                    print(f"FAILED: {result['error']}")
                else:
                    print(f"SUCCESS!")
                    print(f"  Title:   {result.get('title')}")
                    print(f"  Chunks:  {result.get('chunks')}")
                    print(f"  Figures: {result.get('figures')}")
                    print(f"  Tables:  {result.get('tables')}")
            except Exception as e:
                logger.exception("Unexpected error during ingestion of %s", pdf.name)
                print(f"CRITICAL FAILURE: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
