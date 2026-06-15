import asyncio
from sqlalchemy import text
from api.core.database import engine
from src.retrieval.embedder import embed_texts


async def embed_papers():
    print("Embedding papers with SPECTER2...")

    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE papers ADD COLUMN IF NOT EXISTS paper_embedding vector(768)"))
        result = await conn.execute(text("SELECT id, title, abstract FROM papers WHERE paper_embedding IS NULL ORDER BY year DESC NULLS LAST"))
        papers = result.fetchall()

    if not papers:
        print("All papers already embedded.")
        return

    print(f"Embedding {len(papers)} papers...")
    BATCH = 16
    embedded = 0

    for i in range(0, len(papers), BATCH):
        batch = papers[i:i + BATCH]
        texts = [f"{p.title} [SEP] {(p.abstract or '')[:400]}" for p in batch]
        embeddings = embed_texts(texts, batch_size=BATCH)

        async with engine.begin() as conn:
            for paper, emb in zip(batch, embeddings):
                vec = "[" + ",".join(f"{x:.8f}" for x in emb.tolist()) + "]"
                await conn.execute(text(f"UPDATE papers SET paper_embedding = '{vec}'::vector WHERE id = :id"), {"id": paper.id})

        embedded += len(batch)
        print(f"  [{embedded}/{len(papers)}]")

    async with engine.begin() as conn:
        r = await conn.execute(text("SELECT COUNT(*) FROM papers WHERE paper_embedding IS NOT NULL"))
        print(f"Done. {r.scalar()} papers embedded.")

asyncio.run(embed_papers())
