from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from api.core.database import get_db
from api.models.tables import ResearchSession, ChatMessage

router = APIRouter(prefix="/api/v1", tags=["Sessions"])


class SessionCreate(BaseModel):
    title: Optional[str] = "New Session"


class MessageCreate(BaseModel):
    role: str
    content: str


@router.post("/sessions")
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    session = ResearchSession(title=body.title)
    db.add(session)
    await db.flush()
    return {"id": str(session.id), "title": session.title, "created_at": session.created_at}


@router.get("/sessions")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ResearchSession).order_by(desc(ResearchSession.updated_at)).limit(20)
    )
    sessions = result.scalars().all()
    return [{"id": str(s.id), "title": s.title, "created_at": s.created_at, "updated_at": s.updated_at} for s in sessions]


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    )
    msgs = result.scalars().all()
    return [{"id": str(m.id), "role": m.role, "content": m.content, "created_at": m.created_at} for m in msgs]


@router.post("/sessions/{session_id}/messages")
async def add_message(session_id: UUID, body: MessageCreate, db: AsyncSession = Depends(get_db)):
    # verify session exists
    result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    msg = ChatMessage(session_id=session_id, role=body.role, content=body.content)
    db.add(msg)

    # update session title from first user message
    if body.role == "user" and session.title == "New Session":
        session.title = body.content[:60] + ("..." if len(body.content) > 60 else "")

    # bump updated_at
    from datetime import datetime
    session.updated_at = datetime.utcnow()
    await db.flush()
    return {"id": str(msg.id), "role": msg.role, "content": msg.content}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ResearchSession).where(ResearchSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    return {"success": True}
