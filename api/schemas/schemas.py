from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


# ── Query ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_hyde: bool = False
    retrieval_mode: str = Field(default="hybrid")
    session_id: Optional[str] = Field(default="default")


class CitationOut(BaseModel):
    title: str
    authors: str
    year: Optional[int] = None
    section: str
    filename: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    chunks_used: int
    response_type: str
    tokens_used: int
    latency_ms: float
    query_id: UUID


# ── Papers ────────────────────────────────────────────────────────────

class PaperOut(BaseModel):
    id: UUID
    arxiv_id: Optional[str] = None
    filename: str
    title: str
    authors: Optional[str] = None
    abstract: Optional[str] = None
    year: Optional[int] = None
    topic: Optional[str] = None
    url: Optional[str] = None
    chunk_count: int
    indexed: bool
    created_at: datetime

    class Config:
        from_attributes = True


class PaperListResponse(BaseModel):
    papers: list[PaperOut]
    total: int


# ── Search ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=8, ge=1, le=20)
    use_hyde: bool = False


class SearchResultOut(BaseModel):
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    section: str
    score: Optional[float] = None
    text: str
    filename: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultOut]
    total: int


# ── Recommend ─────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class RecommendationOut(BaseModel):
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    filename: str
    similarity: float
    url: Optional[str] = None


class RecommendResponse(BaseModel):
    query: str
    local: list[RecommendationOut]
    arxiv: list[RecommendationOut]


# ── TTS ───────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="Samantha")
    rate: int = Field(default=175, ge=100, le=300)


# ── Flowchart ─────────────────────────────────────────────────────────

class FlowchartRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    explanation: str = Field(..., min_length=10, max_length=5000)


class FlowchartResponse(BaseModel):
    diagram: Optional[str] = None
    success: bool


# ── Feedback ──────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: UUID
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str


# ── Stats ─────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_papers: int
    total_chunks: int
    vector_chunks: int
    embedding_dim: int
    total_queries: int


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    ollama: bool
    index: bool
