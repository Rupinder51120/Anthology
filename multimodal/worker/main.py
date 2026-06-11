"""
Standalone ingestion worker.
Run: python -m multimodal.worker.main --pdf path/to/paper.pdf
"""
import asyncio
import argparse
import os
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://anthology:anthology@localhost:5433/anthology_multimodal"
)


async def setup_db(engine):
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    from multimodal.api.models.tables import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("DB ready.")


async def ingest(pdf_path: str, use_vlm: bool):
    from multimodal.ingestion.pipeline import ingest_paper

    engine = create_async_engine(DATABASE_URL, echo=False)
    await setup_db(engine)

    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    pdf = Path(pdf_path)

    metadata = {
        "filename": pdf.name,
        "title": pdf.stem.replace("_", " "),
    }

    async with Session() as session:
        stats = await ingest_paper(
            pdf_path=str(pdf),
            paper_metadata=metadata,
            session=session,
            use_vlm=use_vlm,
        )

    print(f"\nDone: {stats}")
    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Anthology Multimodal Ingestion Worker")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM figure captioning")
    args = parser.parse_args()

    asyncio.run(ingest(args.pdf, use_vlm=not args.no_vlm))


if __name__ == "__main__":
    main()
