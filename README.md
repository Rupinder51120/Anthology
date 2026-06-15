<div align="center">

<img src="https://img.shields.io/badge/status-active-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/python-3.11-blue?style=flat-square" />
<img src="https://img.shields.io/badge/FastAPI-async-009688?style=flat-square" />
<img src="https://img.shields.io/badge/pgvector-semantic_search-4169E1?style=flat-square" />
<img src="https://img.shields.io/badge/Groq-LLM-F55036?style=flat-square" />
<img src="https://img.shields.io/badge/Langfuse-observability-8B5CF6?style=flat-square" />
<img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" />

# Anthology

### AI Research Intelligence System

*A production RAG platform for citation-grounded question answering over scientific literature.*

**[Live API](https://anthology-api.onrender.com)** · **[Swagger Docs](https://anthology-api.onrender.com/docs)**

</div>

---

## What is this?

Anthology is a full-stack Retrieval-Augmented Generation system that lets researchers ask questions about scientific papers and receive answers grounded in citations.

It is not a wrapper around an LLM. It is an end-to-end information retrieval and generation pipeline — built from scratch — covering ingestion, embedding, hybrid search, reranking, generation, evaluation, and observability.

The system went through five retrieval architectures before reaching its current design:

```
BM25  →  FAISS Dense  →  Hybrid  →  Hybrid + HyDE  →  pgvector + FTS + RRF + Cohere rerank
```

Each iteration was benchmarked on a 100-question evaluation set. The current production stack achieves **Hit@5 = 0.96, MRR = 0.8925**.

---

## System Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                            INGEST PIPELINE                           │
│                                                                      │
│   PDF  ──►  Docling / PyMuPDF                                        │
│               │                                                      │
│               ├──►  Section-aware chunking (title / abstract /       │
│               │     introduction weighted higher in RRF scoring)     │
│               │                                                      │
│               ├──►  Figure captioning  (Groq vision API)             │
│               ├──►  Graph parsing      (DePlot → structured table)   │
│               └──►  SPECTER2 embeddings (768-dim, scientific domain) │
│                               │                                      │
│                               ▼                                      │
│                    PostgreSQL + pgvector                              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     Query arrives     │
                    └───────────┬───────────┘
                                │
               ┌────────────────┴─────────────────┐
               │                                  │
               ▼                                  ▼
      pgvector cosine search            PostgreSQL full-text search
      (SPECTER2 query embed)            (tsvector, ranked by ts_rank)
               │                                  │
               └──────────────┬───────────────────┘
                              │
                              ▼
               Reciprocal Rank Fusion (RRF)
               + section-priority score boost
                              │
                              ▼
               Cohere cross-encoder reranking
               (top-20 → top-5)
                              │
                              ▼
               ┌──────────────────────────────┐
               │       Groq LLM (llama3)      │
               │  citation-grounded prompting │
               │  faithfulness gate           │
               │  conversation memory         │
               └──────────────┬───────────────┘
                              │
                              ▼
               Langfuse — distributed tracing
               retrieve span / rerank span / generate span
```

---


## API

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Full RAG query — returns answer + citations + Langfuse trace ID |
| `GET` | `/api/v1/papers` | List indexed papers |
| `GET` | `/api/v1/papers/{id}` | Paper by ID |
| `POST` | `/api/v1/papers/upload` | Ingest a new PDF |
| `POST` | `/api/v1/search` | Semantic search without generation |
| `POST` | `/api/v1/feedback` | Submit answer feedback |
| `GET` | `/api/v1/stats` | Corpus and system stats |
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/benchmark` | Benchmark results |

**Base URL:** `https://anthology-api.onrender.com`  
**Interactive docs:** `https://anthology-api.onrender.com/docs`

---

## Project Structure

```
Anthology/
│
├── api/
│   ├── core/                   # Pydantic settings, async SQLAlchemy engine
│   ├── models/                 # ORM models — Paper, Chunk, Query, Feedback
│   ├── routers/                # FastAPI route handlers (one file per domain)
│   ├── schemas/                # Pydantic I/O contracts
│   └── services/
│       ├── rag_service.py      # Orchestrates retrieve → rerank → generate → trace
│       ├── ingest_service.py   # PDF → chunks → embeddings → pgvector
│       └── stats_service.py    # Corpus statistics
│
├── src/
│   ├── ingestion/
│   │   ├── parser.py           # Docling / PyMuPDF → ParsedBlock
│   │   ├── chunker.py          # Section-aware chunking with priority weights
│   │   ├── figure_captioner.py # Groq vision → figure captions
│   │   └── graph_parser.py     # DePlot → structured table extraction
│   │
│   ├── retrieval/
│   │   ├── retriever.py        # pgvector + FTS + RRF + Cohere rerank
│   │   ├── embedder.py         # SPECTER2 model loading and caching
│   │   └── hyde.py             # Hypothetical Document Embeddings (optional)
│   │
│   ├── generation/
│   │   ├── generator.py        # Groq / Ollama generation + faithfulness gate
│   │   └── memory.py           # Conversation session management
│   │
│   └── evaluation/
│       ├── evaluator.py        # Hit@k, MRR, nDCG@5
│       ├── benchmarker.py      # QA dataset generation
│       └── pipeline_runner.py  # End-to-end benchmark execution
│
├── frontend/                   # React + Vite + TypeScript
│
├── scripts/
│   ├── embed_papers.py         # Batch embed and sync corpus to pgvector
│   ├── migrate_v2.py           # v2 schema migration
│   └── patch_papers_schema.py
│
├── alembic/                    # Async migrations
├── docker-compose.yml
└── requirements.txt
```

---

## Local Development

**Requirements:** Python 3.11, Docker, Groq API key

```bash
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# set: DATABASE_URL, GROQ_API_KEY, COHERE_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

docker compose up -d          # start PostgreSQL with pgvector
alembic upgrade head          # run migrations
python scripts/embed_papers.py  # embed corpus → pgvector

uvicorn app:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---


## License

MIT
