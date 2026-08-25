# Anthology

A retrieval-augmented generation (RAG) system for question-answering over a self-ingested corpus of research papers, combining hybrid dense/sparse retrieval over PostgreSQL/pgvector with a Python/FastAPI backend and a React/TypeScript frontend.

## Highlights

- Designed and implemented a **hybrid dense + sparse retrieval pipeline** (PostgreSQL/pgvector cosine search + PostgreSQL full-text search, fused with Reciprocal Rank Fusion, reranked with the Cohere API) achieving **77.3% Hit@1 / 85.8% Hit@5 / 0.817 MRR** on a 247-question internal benchmark.
- Built and ran an **external generalization evaluation against the third-party QASPER dataset** (281 papers, 892 questions) via a standalone, reproducible Python script — reporting results transparently, including a case where BM25 outperformed dense embeddings.
- Engineered a **Python/FastAPI backend** (async SQLAlchemy, PostgreSQL, Redis) serving a **React/TypeScript** frontend, with **SSE-streamed, citation-grounded LLM responses** and a swappable Groq/Ollama generation backend.
- Built a **PDF ingestion pipeline** (Docling parsing with OCR, LLM-based table/figure captioning, SPECTER2 embeddings) processing scientific papers end-to-end into a queryable vector store.
- Shipped a **5-service Docker Compose deployment** (API, PostgreSQL, Redis, Ollama, frontend) with **44 passing automated tests** (unit + integration) and a **GitHub Actions CI** pipeline covering 35 of them (see [Testing](#testing) for why the other 9 are excluded).

## Overview

Anthology ingests PDF research papers, parses and chunks them, embeds them, and stores them in PostgreSQL with the `pgvector` extension. At query time it retrieves relevant chunks using a hybrid dense + sparse pipeline, reranks the fused candidates, and streams a cited answer back through the frontend. Papers can be organized into collections, browsed individually with per-paper chat, and new papers can be discovered from ArXiv/OpenAlex. Retrieval quality is evaluated with two separate, checked-in scripts rather than a single quoted number.

This is a solo project run locally against a small (~121-paper) corpus. It is not a hosted product, and this README does not claim otherwise — see [Deployment status](#deployment-status).

## Features

| Feature | Status | Notes |
|---|---|---|
| Hybrid dense + sparse retrieval, RRF fusion, Cohere rerank | **Implemented — live API path** | `hybrid_rerank` is the strategy used by the current API retrieval service; see [Retrieval pipeline](#retrieval-pipeline) |
| Sparse-only, dense-only, dense+HyDE, RRF-without-rerank strategies | **Implemented — benchmark-only** | Reachable only via `scripts/run_benchmark.py`; `RetrievalService` (the live API path) exposes no strategy parameter |
| Cross-encoder local reranking | **Not active** | Model is configured (`api/core/models.py`) and a loader exists in `src/retrieval/retriever.py`, but the code itself comments that it is "retained for future local-rerank fallback; not currently invoked" |
| Content-type metadata filtering (e.g. restrict to tables) | **Implemented, not exposed** | The parameter exists through the retrieval stack, but no router or frontend page currently passes a non-default value |
| Per-paper filtering (chat scoped to one paper) | **Implemented** | `paper_id` is passed through to both the dense and sparse queries and used by the per-paper chat view |
| PDF ingestion: Docling parsing, OCR, section-aware chunking, table/figure captioning | **Implemented** | OCR is enabled in the Docling pipeline config; table/figure chunks get an LLM-generated caption before embedding |
| Streaming answers (SSE) with citations | **Implemented** | Citations are built only from chunks actually included in the prompt |
| Provider-selectable generation (Groq / Ollama) | **Implemented as a static switch** | Not automatic failover — see [Limitations](#limitations) |
| Lexical grounding check | **Implemented, heuristic only** | Word-overlap threshold, not a semantic/entailment model |
| Redis response caching | **Implemented** | |
| Langfuse tracing | **Implemented for both the streaming and non-streaming query paths** | |
| Collections | **Implemented** | |
| ArXiv / OpenAlex paper discovery | **Implemented** | |
| Live benchmark dashboard | **Implemented** | Can trigger a real internal evaluation run from the UI |
| Internal 247-question retrieval benchmark | **Implemented, results below** | |
| External QASPER retrieval evaluation | **Implemented, results below** | Retrieval only, no reranking |
| Generation-quality (faithfulness/relevance/completeness) scoring | **Implemented, but current stored results are unusable** | See [Evaluation](#evaluation) |
| Authentication, rate limiting | **Not implemented** | |
| Public hosted deployment | **Not implemented** | Local Docker only |

## Architecture

```mermaid
flowchart TD
    subgraph CLIENT["React / TypeScript / Vite frontend"]
        UI["Search · Library · Chat · Collections · Discovery · Benchmark"]
    end

    subgraph API["FastAPI backend"]
        ROUTERS["Routers: papers, query, search, discovery,\ncollections, benchmark, sessions, ..."]
        RETRIEVE["Retrieval service"]
        GEN["Generation service"]
    end

    subgraph DATA["Data layer"]
        PG[("PostgreSQL 16 + pgvector\npapers / chunks / collections")]
        REDIS[("Redis\nresponse cache")]
    end

    subgraph EXTERNAL["External services"]
        COHERE["Cohere rerank-v3.5"]
        GROQ["Groq (cloud LLM)"]
        OLLAMA["Ollama (local LLM)"]
        LANGFUSE["Langfuse (tracing)"]
        ARXIV["ArXiv / OpenAlex APIs"]
    end

    subgraph INGEST["Ingestion (per uploaded PDF)"]
        PARSE["Docling parsing + OCR"]
        META["Metadata extraction"]
        CHUNK["Section-aware chunking"]
        ENRICH["Table/figure captioning"]
        EMBED["SPECTER2 embedding"]
    end

    UI <--> ROUTERS
    ROUTERS --> RETRIEVE
    ROUTERS --> GEN
    RETRIEVE -->|dense + sparse search| PG
    RETRIEVE -->|rerank fused candidates| COHERE
    RETRIEVE -.->|cache lookups| REDIS
    GEN -->|USE_GROQ=true| GROQ
    GEN -->|USE_GROQ=false| OLLAMA
    GEN -.->|trace spans| LANGFUSE
    ROUTERS -->|discovery search| ARXIV

    PARSE --> META --> CHUNK --> ENRICH --> EMBED --> PG
    ENRICH -.->|caption generation| GROQ
```

## Retrieval pipeline

**Live API path** — `hybrid_rerank` is the strategy used by the current API retrieval service. `retrieve()`'s `strategy` parameter defaults to `hybrid_rerank`, and the service layer exposed to the API (`RetrievalService`) does not accept a `strategy` argument at all, so a real query through the API goes through all four steps below:

1. **Dense retrieval** — query embedded with the same SPECTER2 model (`allenai/specter2_base`) used at ingestion; PostgreSQL/`pgvector` cosine-similarity search over `chunks.embedding`.
2. **Sparse retrieval** — PostgreSQL `to_tsvector`/`ts_rank_cd` full-text search, computed at query time. No persisted `tsvector` column and no GIN index exist yet (see [Limitations](#limitations)).
3. **Fusion** — the two ranked lists are merged with Reciprocal Rank Fusion.
4. **Reranking** — the fused candidates are reranked by the Cohere `rerank-v3.5` API. If no Cohere key is configured, or the API call fails after 3 retries, this step degrades to the unranked RRF-fused order rather than erroring.

**Benchmark-only strategies** — `sparse`-only, `dense`-only, `dense_hyde` (HyDE query expansion), and `hybrid_rrf` (fusion without rerank) are implemented in `src/retrieval/retriever.py` and are used for the internal benchmark comparison below, but are not reachable through any live API route or frontend page.

**Inactive/reserved code** — a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is configured but its loader is never called anywhere in the codebase. `content_type` metadata filtering exists in the retrieval functions' signatures but has no current caller passing a real value; `paper_id` filtering, by contrast, is genuinely used (per-paper chat).

## Ingestion pipeline

```
PDF upload
  → Docling parsing (OCR enabled in the pipeline configuration; Docling applies it per-page as needed)
  → metadata extraction (title/authors/year — heuristic, not 100% complete)
  → section-aware chunking
  → table/figure captioning (LLM-generated, via Groq)
  → SPECTER2 embedding (768-dim)
  → write to PostgreSQL (papers, chunks tables)
```

Ingestion is transactional per paper: a failure partway through does not leave a partial row. Upload is restricted to PDF files, 50MB max, with filename sanitization against path traversal (see [Security](#security)). This processes one paper per API request — there is no batch ingestion queue, and it has been exercised against 121 real papers, not at production scale.

## Evaluation

Two evaluations exist, and they measure different things.

### 1. Internal benchmark — strategy comparison on the ingested corpus

247 questions, generated from the actual ingested corpus via local Ollama (a question is generated from a chunk; retrieval is then scored on whether it finds that chunk back). This measures which retrieval strategy performs best **on this corpus**, using a **self-referential** question set — it does not measure generalization to unseen data or question styles.

### 2. External evaluation — QASPER (Dasigi et al., 2021)

[QASPER](https://huggingface.co/datasets/allenai/qasper) is a third-party, CC-BY-4.0 dataset: 5,049 questions over 1,585 NLP papers, each written by someone who had read only the paper's title and abstract. `scripts/evaluate_qasper.py` loads the dataset directly (full paper text ships with it — no PDF download needed), builds a separate, in-memory index that never touches the production database or corpus, and evaluates **retrieval only** (dense, sparse/BM25, hybrid RRF). No Cohere reranking and no generation/answer-quality scoring were run against QASPER — the script does not implement either, so no claim is made about them.

## Verified results

### Internal benchmark — hybrid_rerank, full 121-paper corpus, 247 questions

| Metric | Value |
|---|---|
| Paper Hit@1 | 0.773 |
| Paper Hit@5 | 0.858 |
| Paper MRR | 0.817 |

`hybrid_rerank` is the production default and the highest-scoring strategy of the six compared internally. These numbers describe strategy comparison **on this corpus only** — they are not an external or generalization benchmark, and no generation-quality (faithfulness/relevance/completeness) numbers are reported here: the stored judge scores from the last full run are a flat ~0.50 across all three metrics, a known artifact of a Groq daily-quota exhaustion mid-run, not a real measurement.

### External evaluation — QASPER validation split, 281 papers, 892 answerable questions, retrieval only

| Strategy | Paper Hit@1 | Paper Hit@5 | Paper MRR |
|---|---|---|---|
| dense (SPECTER2) | 0.128 | 0.230 | 0.178 |
| sparse (BM25) | 0.221 | 0.353 | 0.281 |
| hybrid_rrf | 0.197 | 0.370 | 0.272 |

Stated plainly: **BM25 lexical search outperformed the SPECTER2 dense embedding on this evaluation**, the reverse of the internal ranking, and both are well below the internal benchmark's numbers. This does not show that BM25 is universally better, or that SPECTER2 is a poor embedding choice in general — it's a single result on one external dataset with a specific question style (QASPER questions are derived only from a paper's title/abstract). It is reported here because it's what was actually measured, not because it's flattering.

## Current corpus

| Metric | Value |
|---|---|
| Papers | 121 |
| Chunks | 11,889 |
| Embedded chunks | 11,889 / 11,889 |
| Orphan chunks | 0 |

This is the **current** state of the local corpus, verified via `GET /api/v1/stats` against a running instance — not a fixed or guaranteed size. One exact-duplicate paper (identical PDF, identical extracted text) was identified and removed during a repository cleanup pass; the numbers above reflect the corpus after that removal. No PDFs or paper content are included in this repository (`data/papers/` is gitignored).

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| API | Python, FastAPI | REST API routing, request/response validation |
| Database | PostgreSQL 16 + pgvector | Paper/chunk storage, vector search, full-text search |
| ORM / migrations | SQLAlchemy (async) + Alembic | |
| Cache | Redis | Response caching |
| Embeddings | SPECTER2 (`sentence-transformers`) | Scientific-paper-tuned dense embeddings |
| PDF parsing | Docling | Layout-aware parsing with OCR |
| Reranking | Cohere `rerank-v3.5` | |
| Generation (cloud) | Groq | Used when `USE_GROQ=true` |
| Generation (local) | Ollama | Used when `USE_GROQ=false` |
| Tracing | Langfuse | |
| Frontend | React + TypeScript + Vite | |
| Containerization | Docker Compose | 5-service local stack |
| CI | GitHub Actions | |
| Testing | pytest, pytest-asyncio | |
| External APIs | ArXiv, OpenAlex, HuggingFace `datasets` | Discovery feature; QASPER evaluation |

## Project structure

```
anthology/
├── api/                    # FastAPI app
│   ├── routers/              # papers, query, search, discovery, collections, benchmark, ...
│   ├── services/              # rag_service, paper_service, ingest_service
│   ├── models/                # SQLAlchemy ORM tables
│   └── core/                  # config, database engine, model-name registry
├── src/
│   ├── ingestion/              # Docling parser, chunker, metadata_resolver
│   ├── retrieval/               # embedder (SPECTER2), retriever (hybrid pipeline)
│   ├── generation/               # Groq/Ollama generator, streaming + non-streaming
│   └── evaluation/               # evaluator, benchmarker
├── scripts/
│   ├── run_benchmark.py           # internal 6-strategy benchmark
│   └── evaluate_qasper.py         # external QASPER retrieval evaluation
├── benchmarks/qa_dataset_v1.json   # the 247-question internal dataset
├── alembic/                 # database migrations
├── frontend/                # React/TypeScript/Vite app
├── tests/                   # pytest suite
├── .github/workflows/        # CI
├── docker-compose.yml
└── .dockerignore / .env.example
```

`docs/` contains internal audit notes used during development; those files are intentionally excluded from version control (`.gitignore`) and are not part of this repository's public contents.

## Setup

```bash
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology
cp .env.example .env   # fill in values as needed — see Environment variables below
```

## Environment variables

Copy `.env.example` and fill in real values locally; never commit them. Placeholders only, below:

| Variable | Required? | Purpose |
|---|---|---|
| `DATABASE_URL` | Required | `postgresql+asyncpg://<user>:<password>@<host>:5432/<dbname>` |
| `REDIS_URL` | Required | e.g. `redis://<host>:6379` |
| `USE_GROQ` | Optional (default `false`) | `true` = Groq for generation; `false` = Ollama. A config switch, not failover. |
| `GROQ_API_KEY` | Required only if `USE_GROQ=true` | |
| `COHERE_API_KEY` | Required for reranking | Without it, reranking degrades to unranked RRF order rather than failing |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Optional | Tracing only; app runs without them |
| `OLLAMA_URL` | Optional (default `http://localhost:11434`) | |

The `DATABASE_URL` shown in `.env.example` (`postgresql+asyncpg://anthology:anthology@localhost:5432/anthology`) is a local development default for the Docker Compose Postgres container, not a production credential.

## Docker

```bash
docker compose up -d
```

Starts 5 containers, per `docker-compose.yml`: `api` (FastAPI, port 8000), `db` (Postgres 16 + pgvector, port 5432), `redis` (port 6379), `ollama` (port 11434), `frontend` (React, port 5173). `.dockerignore` (root and `frontend/`) excludes secrets, local datasets, and internal docs from the build context.

On a genuinely fresh database, the API creates its schema automatically at startup. Against a pre-existing database from an earlier version, apply migrations explicitly:

```bash
docker compose exec api alembic upgrade head
```

Health check: `curl http://localhost:8000/health`

## Testing

```bash
pytest tests/
```

44 of 44 tests pass locally. CI (`.github/workflows/tests.yml`) runs 35 of those 44 — the other 9 are excluded for documented reasons, not silently dropped:
- 1 test (`test_collections_add_paper.py`) is a live integration test requiring a running, corpus-seeded PostgreSQL instance, which the CI environment does not provision.
- 8 tests (`test_retrieval_alignment.py`) mock the embedding call itself, but importing the embedder module still pulls in the full `sentence-transformers`/`torch` stack at import time — installing that in CI for tests whose logic never touches the model wasn't judged worthwhile.

No code-coverage tooling is configured, so no coverage percentage is claimed.

## Evaluation / benchmark commands

```bash
# Internal 6-strategy benchmark (writes indexes/results_*.json)
python scripts/run_benchmark.py

# External QASPER evaluation (downloads the dataset from HuggingFace on first run)
python scripts/evaluate_qasper.py --num-papers 40            # bounded pilot
python scripts/evaluate_qasper.py --num-papers 281 --split validation   # full validation split (what the numbers above are from)
```

## Limitations

- **No automatic generation-provider failover.** `USE_GROQ` is a static switch. If the selected provider's call fails, the request returns/streams an error — it does not retry the other provider.
- **Grounding is a lexical heuristic**, not a semantic or entailment-based check.
- **The internal benchmark is self-referential** and does not by itself demonstrate generalization; the QASPER evaluation is the external check, and its results (BM25 outperforming dense retrieval) should be read as a real, dataset-specific finding, not glossed over.
- **Generation-quality judge scores for the internal benchmark are currently unusable** (Groq quota exhaustion mid-run produced flat, uninformative values).
- **Metadata extraction is heuristic-based** — title/author extraction is largely complete, but fields like `arxiv_id`, `doi`, `abstract`, and `year` are missing for a meaningful fraction of papers. Doesn't affect retrieval, only metadata display.
- **No authentication, no rate limiting**, on any route.
- **No ANN or GIN indexing yet** — acceptable at the current ~12K-chunk scale; would need attention well before this reaches the hundreds of thousands of chunks.
- **Cohere reranking runs on a trial-tier key** (rate-limited); repeated failures fall back to unranked ordering.
- **Ollama runs locally**, so generation speed/quality with `USE_GROQ=false` depends on the machine it's running on, not a managed service.
- **Not exercised at scale** — everything above describes a ~121-paper corpus; behavior at 10x or 100x that size is untested.

## Security

Verified in place:
- **CORS restricted** to `http://localhost:3000` and `http://localhost:5173` — not wildcarded.
- **Path traversal protection on upload**: uploaded filenames are reduced to a safe basename and the resolved destination path is checked to be contained within the upload directory before writing.
- **SQL injection protection**: query construction uses either SQLAlchemy's ORM query builder (`select()`) or `text()` with bound parameters (`:param`-style); no raw string interpolation of user input into SQL was found in the routes checked.
- **Secrets read from environment variables**, not hardcoded.

Explicitly not in place:
- No authentication or authorization on any route.
- No rate limiting on any route, including upload and the benchmark-trigger endpoint.
- No comprehensive security audit has been performed — the items above are specific, verified protections, not a general security guarantee.

## Deployment status

**There is currently no public hosted deployment of Anthology.** It runs locally via Docker Compose, as described above. Making it publicly reachable would additionally require, at minimum: a hosted PostgreSQL/pgvector instance, hosted (not locally-run) generation inference, authentication, rate limiting, production secret management, and appropriate indexing for scale beyond the current corpus size — none of which are currently implemented.

## Future improvements

- A clean generation-quality judge run once Groq quota/cost for it is available (the current one is unusable).
- Extend the QASPER evaluation with Cohere reranking, to compare directly against the internal `hybrid_rerank` default rather than only dense/sparse/hybrid_rrf.
- GIN index (full-text search) and an ANN index (vectors) once the corpus grows past the current scale.
- Authentication and rate limiting, required before any public exposure.

## License

MIT — see [LICENSE](LICENSE).
