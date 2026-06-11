import uuid
from pgvector.sqlalchemy import Vector
from datetime import datetime
from sqlalchemy import String, Text, Float, Integer, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    figure_count: Mapped[int] = mapped_column(Integer, default=0)
    table_count: Mapped[int] = mapped_column(Integer, default=0)
    indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="paper")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)

    # Modality
    content_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    # text | table | figure | equation | caption

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # For figures: VLM-generated caption + description
    # For tables: markdown + generated summary
    # For text: chunk text
    # For equations: LaTeX + text description

    # Location
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    figure_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # e.g. "Figure 3", "Table 2", "Equation 1"

    # Figure-specific
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # local path to cropped figure image

    # Table-specific
    table_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    table_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Embedding
    embedding: Mapped[list | None] = mapped_column(Vector(768), nullable=True)

    # Quality
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Raw metadata
    extra_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    paper: Mapped["Paper"] = relationship(back_populates="chunks")
