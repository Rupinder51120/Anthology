# Anthology

## AI Research Intelligence Platform

Anthology is a multimodal research intelligence system for scientific literature.

Unlike traditional "chat with PDF" applications, Anthology combines citation-aware embeddings, hybrid retrieval, reranking, multimodal document understanding, evaluation, and observability to provide grounded answers over research papers.

The system processes text, figures, charts, and tables, retrieves supporting evidence through a multi-stage retrieval pipeline, and generates citation-backed responses using large language models.

---

## Key Features

### Retrieval

* Citation-aware scientific embeddings using SPECTER2
* Hybrid retrieval:

  * PostgreSQL Full-Text Search (FTS)
  * pgvector semantic search
* Reciprocal Rank Fusion (RRF)
* Cohere cross-encoder reranking
* HyDE query expansion

### Multimodal Understanding

* PDF parsing with Docling and PyMuPDF
* Figure captioning using vision-language models
* Table extraction and summarization
* Structured graph and chart understanding
* Section-aware chunking

### Generation

* Citation-grounded answer generation
* Multi-turn conversational memory
* Faithfulness verification
* Source attribution

### Evaluation

* Paper-level retrieval metrics
* Chunk-level retrieval metrics
* Hit@K
* MRR
* nDCG
* Faithfulness scoring
* Relevance scoring
* Completeness scoring

### Observability

* End-to-end Langfuse tracing
* Retrieval span tracking
* Reranking span tracking
* Generation span tracking

---

# System Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           INGESTION PIPELINE                         │
└──────────────────────────────────────────────────────────────────────┘

PDF
 │
 ▼
Docling / PyMuPDF
 │
 ├── Text Extraction
 ├── Figure Extraction
 ├── Table Extraction
 └── Metadata Extraction
 │
 ▼
Section-Aware Chunking
 │
 ├── Abstract
 ├── Introduction
 ├── Methods
 ├── Results
 └── Conclusion
 │
 ▼
Multimodal Enrichment
 │
 ├── Figure Captioning
 ├── Table Summarization
 └── Graph Parsing
 │
 ▼
SPECTER2 Embeddings
 │
 ▼
PostgreSQL + pgvector
```

```text
┌──────────────────────────────────────────────────────────────────────┐
│                             QUERY PIPELINE                           │
└──────────────────────────────────────────────────────────────────────┘

User Query
 │
 ▼
HyDE Query Expansion (optional)
 │
 ▼
SPECTER2 Query Embedding
 │
 ├─────────────────────┐
 ▼                     ▼

pgvector Search     PostgreSQL FTS
 │                     │
 └──────────┬──────────┘
            ▼

Reciprocal Rank Fusion
(RRF)

            ▼

Section-Aware Boosting

            ▼

Cohere Rerank v3.5

            ▼

Top Evidence Chunks

            ▼

Groq / Ollama LLM

            ▼

Answer + Citations

            ▼

Langfuse Tracing
```

---

# Retrieval Pipeline

Anthology uses a multi-stage retrieval architecture optimized for scientific literature.

1. Query embedding with SPECTER2
2. Semantic retrieval via pgvector
3. Lexical retrieval via PostgreSQL Full-Text Search
4. Reciprocal Rank Fusion (RRF)
5. Section-aware ranking
6. Cohere cross-encoder reranking
7. Context assembly
8. Citation-grounded generation

This design improves retrieval robustness for both keyword-driven and conceptual research questions.

---

# Evaluation Framework

Anthology includes a built-in benchmarking framework for evaluating both retrieval quality and answer quality.

## Retrieval Metrics

### Paper-Level

* Hit@K
* MRR
* nDCG

### Chunk-Level

* Hit@K
* MRR
* nDCG

Chunk-level evaluation verifies whether the specific evidence chunk was retrieved, not merely whether the correct paper appeared in results.

## Generation Metrics

* Faithfulness
* Relevance
* Completeness

## Benchmark Dataset Generation

Benchmark datasets are automatically generated from research papers while minimizing lexical leakage between questions and source evidence.

This enables retrieval evaluation and generation evaluation to be measured independently.

---

# API

| Method | Route                   | Description           |
| ------ | ----------------------- | --------------------- |
| POST   | `/api/v1/query`         | Full RAG query        |
| POST   | `/api/v1/search`        | Retrieval only        |
| POST   | `/api/v1/papers/upload` | Upload and ingest PDF |
| GET    | `/api/v1/papers`        | List indexed papers   |
| GET    | `/api/v1/papers/{id}`   | Get paper             |
| POST   | `/api/v1/feedback`      | Submit feedback       |
| GET    | `/api/v1/stats`         | Corpus statistics     |
| GET    | `/api/v1/benchmark`     | Benchmark results     |
| GET    | `/health`               | Health check          |

### Production API

https://anthology-api.onrender.com

### API Documentation

https://anthology-api.onrender.com/docs

---

# Tech Stack

## Backend

* FastAPI
* Async SQLAlchemy
* Alembic
* Pydantic

## Database

* PostgreSQL
* pgvector
* PostgreSQL Full-Text Search

## Retrieval

* AllenAI SPECTER2
* Reciprocal Rank Fusion
* Cohere Rerank v3.5
* HyDE

## Multimodal Processing

* Docling
* PyMuPDF
* Vision Models
* Table Summarization Pipeline

## LLMs

* Groq
* Llama 3
* Ollama
* Qwen 2.5

## Observability

* Langfuse

## Frontend

* React
* TypeScript
* Vite

---

# Project Structure

```text
Anthology/
│
├── api/
│   ├── core/
│   ├── models/
│   ├── routers/
│   ├── schemas/
│   └── services/
│
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
│
├── frontend/
├── scripts/
├── alembic/
├── docker-compose.yml
└── requirements.txt
```

---

# Local Development

## Requirements

* Python 3.11+
* Docker
* PostgreSQL
* Groq API Key
* Cohere API Key

```bash
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
```

Configure:

```env
DATABASE_URL=
GROQ_API_KEY=
COHERE_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

Start services:

```bash
docker compose up -d

alembic upgrade head

python scripts/embed_papers.py

uvicorn app:app --reload
```

API:

```text
http://localhost:8000
```

Docs:

```text
http://localhost:8000/docs
```

---

# Future Roadmap

* Research paper discovery
* Semantic Scholar integration
* ArXiv integration
* Multi-paper comparison
* Research trend analysis
* Agentic literature review workflows
* Citation graph exploration
* Research recommendation engine

---

# License

MIT License
