from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from api.core.database import get_db
from api.models.tables import Collection, CollectionPaper, Paper

router = APIRouter(prefix="/api/v1", tags=["Collections"])


class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    color: Optional[str] = "#7c5cfc"


@router.get("/collections")
async def list_collections(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Collection).order_by(desc(Collection.updated_at)))
    cols = result.scalars().all()
    out = []
    for c in cols:
        count_r = await db.execute(
            select(func.count()).where(CollectionPaper.collection_id == c.id)
        )
        out.append({
            "id": str(c.id), "name": c.name, "description": c.description,
            "color": c.color, "paper_count": count_r.scalar(), "created_at": c.created_at,
        })
    return out


@router.post("/collections")
async def create_collection(body: CollectionCreate, db: AsyncSession = Depends(get_db)):
    col = Collection(name=body.name, description=body.description, color=body.color)
    db.add(col)
    await db.flush()
    return {"id": str(col.id), "name": col.name, "description": col.description, "color": col.color, "paper_count": 0}


@router.delete("/collections/{col_id}")
async def delete_collection(col_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Collection).where(Collection.id == col_id))
    col = result.scalar_one_or_none()
    if not col:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(col)
    return {"success": True}


@router.get("/collections/{col_id}/papers")
async def get_collection_papers(col_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Paper).join(CollectionPaper, CollectionPaper.paper_id == Paper.id)
        .where(CollectionPaper.collection_id == col_id)
    )
    papers = result.scalars().all()
    return [{"id": str(p.id), "title": p.title, "authors": p.authors, "year": p.year, "filename": p.filename} for p in papers]


@router.post("/collections/{col_id}/papers/{paper_id}")
async def add_paper(col_id: UUID, paper_id: UUID, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(CollectionPaper).where(
            CollectionPaper.collection_id == col_id,
            CollectionPaper.paper_id == paper_id,
        )
    )
    if existing.scalar_one_or_none():
        return {"success": True}
    db.add(CollectionPaper(collection_id=col_id, paper_id=paper_id))
    return {"success": True}


@router.delete("/collections/{col_id}/papers/{paper_id}")
async def remove_paper(col_id: UUID, paper_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(CollectionPaper).where(
            CollectionPaper.collection_id == col_id,
            CollectionPaper.paper_id == paper_id,
        )
    )
    link = result.scalar_one_or_none()
    if link:
        await db.delete(link)
    return {"success": True}
