# Anthology

A retrieval-augmented question-answering system over a self-ingested corpus of research papers: a FastAPI backend with a hybrid dense+sparse retrieval pipeline (PostgreSQL/pgvector + Cohere reranking), a streaming-answer React frontend, and two separate, reproducible evaluation harnesses — one internal, one against a third-party dataset.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Retrieval pipeline](#retrieval-pipeline)
- [Generation](#generation)
- [Ingestion pipeline](#ingestion-pipeline)
- [Current corpus](#current-corpus)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Local setup / Docker](#local-setup--docker)
- [Environment variables](#environment-variables)
- [Deployment status](#deployment-status)
- [Limitations](#limitations)
- [Security](#security)
- [Project structure](#project-structure)
- [Technology stack](#technology-stack)
- [Reproducing everything in this README](#reproducing-everything-in-this-readme)
- [Why this project is technically interesting](#why-this-project-is-technically-interesting)

---

## Overview

**What it does.** Anthology ingests PDF research papers, parses and chunks them, embeds them, and stores them in Postgres with `pgvector`. At query time it retrieves relevant chunks using a hybrid dense+sparse pipeline, reranks them, and streams a cited answer back through a React frontend. It also organizes papers into collections, surfaces new papers from ArXiv/OpenAlex, and includes two separate, checked-in evaluation scripts rather than a single quoted accuracy number.

**Problem it addresses.** Reading and cross-referencing a stack of PDFs by hand doesn't scale past a handful of papers. Anthology lets you point it at a folder of PDFs and then ask questions across the whole set, get answers grounded in the actual text with citations back to source chunks, and search/organize the corpus instead of re-reading it.

**What's technically interesting about it**, in order of how much engineering went into each:
- A hybrid retrieval pipeline (dense pgvector search + Postgres full-text search, fused with Reciprocal Rank Fusion, then reranked) that is actually benchmarked, not assumed to be an improvement.
- Two independently-run evaluation harnesses — an internal 247-question strategy comparison and an external evaluation against the real [QASPER](https://huggingface.co/datasets/allenai/qasper) dataset — with the external one's results reported honestly even where they contradict the internal ranking (see [Evaluation](#evaluation)).
- A streaming generation path (SSE) that shares its provider-selection and prompt-construction logic with the non-streaming path, rather than duplicating it (this was previously a real bug in the codebase — see [Limitations](#limitations)).
- A CI workflow whose dependency list and test exclusions were verified by actually running it in an isolated clean virtual environment before being committed, not just written and assumed to work.

This is a solo project, not a team-maintained production service. Sections below are explicit about what's implemented, what's local-only/experimental, and what's simply absent.

## Features

| Feature | Status | Evidence |
|---|---|---|
| Hybrid dense+sparse retrieval with RRF fusion + Cohere reranking | **Implemented, production path** | `src/retrieval/retriever.py` — this is the only strategy the live API exposes (`strategy` defaults to `hybrid_rerank`; no endpoint or frontend code passes a different value) |
| Alternative retrieval strategies (sparse-only, dense-only, dense+HyDE, RRF-without-rerank, dense+rerank) | **Implemented, benchmark-only** | Same file, `VALID_STRATEGIES` — only reachable through `scripts/run_benchmark.py`, not through any live API route |
| Per-paper query scoping (chat about one specific paper) | **Implemented** | `paper_id` filter threaded through `RetrievalService` → `pgvector_search`/`postgres_fts_search`, used by the frontend's per-paper chat view |
| Content-type metadata filtering (e.g. tables vs. text) | **Implemented in the retriever, not exposed** | `content_type` parameter exists in `retrieve()`/`pgvector_search()`/`postgres_fts_search()`, but no API router or frontend page currently passes a non-`None` value — it's live plumbing with no caller |
| Cross-encoder local reranking | **Not active (reserved for future use)** | `src/retrieval/retriever.py` explicitly comments `_cross_encoder = None # retained for future local-rerank fallback; not currently invoked` |
| Streaming SSE answers with inline citations | **Implemented** | `api/routers/query.py` `/query/stream`, backed by `stream_answer()` in `src/generation/generator.py` |
| Swappable generation provider (Groq cloud / Ollama local) | **Implemented as a static config switch, not automatic failover** | `_groq_enabled()` reads `settings.use_groq`; a Groq call failure returns an error response, it does not retry against Ollama |
| Lexical grounding check on generated answers | **Implemented, heuristic-only** | `_is_grounded()` — requires >15% of the answer's >4-character words to appear in the retrieved context; not a semantic/entailment model |
| Multimodal PDF ingestion (Docling parsing, OCR, table/figure LLM captioning) | **Implemented** | `src/ingestion/parser.py` sets `pipeline_options.do_ocr = True`; table/figure chunks get Groq-generated captions before embedding |
| Response caching | **Implemented** | Redis-backed cache in `api/services/rag_service.py` |
| Request tracing/observability | **Implemented for the non-streaming and streaming query paths** | Langfuse spans in `api/services/rag_service.py` and `api/routers/query.py` |
| Collections (organize papers into named groups) | **Implemented** | `api/routers/collections.py`; add/remove persistence was a confirmed, fixed bug — see [Limitations](#limitations) |
| Paper discovery from ArXiv + OpenAlex | **Implemented** | `api/routers/discovery.py`, frontend `Discovery.tsx` |
| Live benchmark dashboard (trigger a real internal eval from the UI) | **Implemented** | `api/routers/benchmark.py`, frontend `Benchmark.tsx` |
| Internal 247-question retrieval benchmark | **Implemented, real results** | `benchmarks/qa_dataset_v1.json` + `scripts/run_benchmark.py` — see [Evaluation](#evaluation) |
| External QASPER retrieval-generalization evaluation | **Implemented, real results, retrieval-only** | `scripts/evaluate_qasper.py` — see [Evaluation](#evaluation) |
| Generation-quality (faithfulness/relevance/completeness) scoring | **Implemented but currently contaminated** | The stored judge scores from the last full run are a flat ~0.50 across all three metrics due to a Groq daily-quota exhaustion mid-run; not usable as evidence of anything |
| Authentication / authorization | **Not implemented** | No auth dependency exists anywhere in `api/` |
| Rate limiting | **Not implemented** | No rate-limiting middleware/library present |
| Public hosted deployment | **Not implemented** | Local Docker Compose only — see [Deployment status](#deployment-status) |

## Architecture

```mermaid
flowchart TD
    subgraph CLIENT["React Frontend (Vite/TS)"]
        UI["Search / Library / Chat / Collections / Discovery / Benchmark pages"]
    end

    subgraph API["FastAPI Backend"]
        ROUTERS["Routers: papers, query, search, discovery,\ncollections, benchmark, sessions, ..."]
        RETRIEVE["Retrieval Service"]
        GEN["Generation Service"]
    end

    subgraph DATA["Data Layer"]
        PG[("PostgreSQL 16 + pgvector\npapers / chunks / collections")]
        REDIS[("Redis\nresponse cache")]
    end

    subgraph EXTERNAL["External Services"]
        COHERE["Cohere rerank-v3.5"]
        GROQ["Groq (cloud LLM)"]
        OLLAMA["Ollama (local LLM)"]
        LANGFUSE["Langfuse (tracing)"]
        ARXIV["ArXiv / OpenAlex\n(paper discovery)"]
    end

    subgraph INGEST["Ingestion (offline, per uploaded PDF)"]
        PARSE["Docling parsing + OCR"]
        META["Metadata extraction"]
        CHUNK["Section-aware chunking"]
        ENRICH["Table/figure captioning (Groq)"]
        EMBED["SPECTER2 embedding"]
    end

    UI <--> ROUTERS
    ROUTERS --> RETRIEVE
    ROUTERS --> GEN
    RETRIEVE -->|dense + sparse search| PG
    RETRIEVE -->|rerank fused candidates| COHERE
    RETRIEVE -.->|cached responses| REDIS
    GEN -->|USE_GROQ=true| GROQ
    GEN -->|USE_GROQ=false| OLLAMA
    GEN -.->|trace spans| LANGFUSE
    ROUTERS -->|live discovery search| ARXIV

    PARSE --> META --> CHUNK --> ENRICH --> EMBED --> PG
    ENRICH -.->|caption generation| GROQ
```

## Retrieval pipeline

The **only strategy the live API ever executes** is `hybrid_rerank` — `retrieve()`'s `strategy` parameter defaults to it, and neither `RetrievalService` nor any router passes a different value. The other five strategies below exist and are real, but are reachable only through `scripts/run_benchmark.py` for comparison purposes.

1. **Dense retrieval** (`pgvector_search`) — the query is embedded with the same SPECTER2 model used at ingestion time (`allenai/specter2_base`, 768-dim); Postgres/`pgvector` performs cosine-similarity search over `chunks.embedding` via the `<=>` operator.
2. **Sparse retrieval** (`postgres_fts_search`) — Postgres `to_tsvector`/`ts_rank_cd` full-text search, computed at query time. There is no persisted `tsvector` column and no GIN index — acceptable at the current corpus size (see [Limitations](#limitations)), but a sequential-scan cost that would need addressing at a larger scale.
3. **Fusion** (`rrf_fuse`) — the two ranked candidate lists are merged via Reciprocal Rank Fusion (`RRF_K = 60`).
4. **Reranking** (`rerank`) — the fused candidates are reranked by the Cohere `rerank-v3.5` API. This is the only reranking method actually invoked in the code path; a local cross-encoder model is configured (`cross-encoder/ms-marco-MiniLM-L-6-v2`) but its loader function is never called anywhere (see [Features](#features)).
5. **Metadata filtering** — `paper_id` filtering is real and used (per-paper chat scoping). `content_type` filtering is implemented at the same level but has no current caller.
6. **HyDE (Hypothetical Document Embeddings)** — implemented (`dense_hyde` strategy, `_embed_query_hyde`), but excluded from the production path and only benchmarked on a 10-question sample (too small to report a result from) due to its latency (~23x normal query latency, dominated by the local Ollama generation step it requires).

## Generation

- **Providers**: Groq (cloud, default model `openai/gpt-oss-20b`) or Ollama (local, default model `qwen2.5:7b`). Selection is a single static boolean, `settings.use_groq` (`USE_GROQ` env var) — read once at request time via `_groq_enabled()`.
- **This is a configuration-selected provider, not automatic failover.** If the selected provider's call raises an exception (rate limit, network error, bad auth), `generate_answer()`/`stream_answer()` catch it and return/yield a user-facing error — there is no retry against the other provider.
- **Streaming**: `/query/stream` (SSE) calls the same `stream_answer()` function that respects the same provider switch, prompt construction (`_build_messages`), and citation formatting (`format_citations`) as the non-streaming `/query` endpoint — they were previously two separate, inconsistent implementations; this was fixed so there is a single source of truth for both paths.
- **Citations**: built directly from the chunks actually included in the prompt (`format_citations(used_chunks)`), so a citation structurally cannot reference a chunk that wasn't part of the context the model saw.
- **Grounding**: a lexical heuristic (`_is_grounded`), not a semantic/entailment check — see [Features](#features) for the exact threshold.

## Ingestion pipeline

```
PDF upload
  → Docling parsing (OCR enabled for image-only pages/PDFs)
  → metadata extraction (title/authors/year/etc., heuristic-based)
  → section-aware chunking (RecursiveCharacterTextSplitter, tuned to preserve
     short high-value scientific facts and table/figure boundaries)
  → table/figure enrichment (Groq-generated captions/summaries for non-text chunks)
  → SPECTER2 embedding (768-dim)
  → write to Postgres (papers, chunks tables; pgvector extension)
```

- Ingestion is per-paper transactional: a failure partway through (e.g., a timeout) does not leave a partial row.
- Upload is limited to PDF files, 50MB max (enforced server-side at two call sites: direct upload and fetch-by-URL).
- Filenames are sanitized against path traversal before being used to construct a filesystem path (`_safe_pdf_filename`, plus a resolved-path containment check as defense in depth).
- This is **not** a production-scale ingestion pipeline: it processes one paper at a time via a single API request, with no batching, queueing, or horizontal scaling. It has been run against 121 real papers, not thousands.

## Current corpus

As of the last verification pass:

| Metric | Value |
|---|---|
| Papers | 121 |
| Chunks | 11,889 |
| Embedded chunks | 11,889 / 11,889 (100%) |
| Orphan chunks | 0 |

**These numbers describe the current local/test corpus, not a fixed or guaranteed size.** The corpus grows or shrinks as papers are uploaded or removed through the app; re-run `curl http://localhost:8000/api/v1/stats` against a running instance for the current count. No PDFs or paper content are included in this repository — `data/papers/` is gitignored.

## Evaluation

Anthology is evaluated two ways. **These measure different things and are not interchangeable.**

### A. Internal benchmark — strategy comparison on the actual corpus

247 questions, self-generated from the ingested corpus via local Ollama (a question is generated from a chunk, then retrieval is scored on whether it finds that chunk back). This answers *"which retrieval strategy works best on our own data?"* — it is self-referential by construction and structurally favors lexically-similar strategies, so it does not answer *"how well does this generalize?"*

Full-corpus results (`indexes/results_hybrid_rerank_fixed_hybrid_rerank_7_scores.json`), n=247:

| Metric | Value |
|---|---|
| Paper Hit@1 | 77.3% |
| Paper Hit@5 | 85.8% |
| Paper MRR | 81.7% |
| Chunk Hit@1 | 57.5% |
| Chunk Hit@5 | 72.5% |
| Chunk MRR | 64.3% |

Hit@k = the fraction of questions where the source paper (or exact source chunk, for the chunk-level numbers) appeared in the top-k retrieved results. MRR = mean reciprocal rank of the first correct hit. This is the production `hybrid_rerank` strategy; five other strategies (sparse, dense, dense+HyDE, hybrid without rerank, dense+rerank) were also benchmarked for comparison — `hybrid_rerank` scored highest on every metric among the strategies with full-corpus coverage.

### B. External benchmark — generalization check against QASPER

[QASPER](https://huggingface.co/datasets/allenai/qasper) (Dasigi et al., 2021) is a third-party, CC-BY-4.0-licensed dataset: 5,049 questions over 1,585 NLP papers, each written by a practitioner who had read only the paper's title and abstract. `scripts/evaluate_qasper.py` loads it directly (full paper text ships in the dataset — no PDF download needed), builds a completely separate in-memory index (this never touches the production Postgres corpus or database), and evaluates **retrieval only** — dense (Anthology's real SPECTER2 embedder), sparse (BM25), and hybrid RRF fusion. No Cohere reranking and no generation/answer-quality evaluation are performed here.

Measured results, full validation split, 281 papers, 892 answerable questions:

| Strategy | Paper Hit@1 | Paper Hit@5 | Paper MRR |
|---|---|---|---|
| dense (SPECTER2) | 12.8% | 23.0% | 17.8% |
| sparse (BM25) | 22.1% | 35.3% | 28.1% |
| hybrid_rrf | 19.7% | 37.0% | 27.2% |

**Reported as-is, not cherry-picked**: retrieval scores are substantially lower than the internal benchmark's, and **BM25 lexical search outperforms the SPECTER2 dense embedding here** — the opposite of the internal ranking. This is not evidence that BM25 is universally superior, or that SPECTER2 is a poor embedding choice in general; it's an evaluation/generalization finding specific to this dataset and question style (QASPER questions are derived only from a paper's title/abstract, which may match differently against BM25's lexical overlap than against SPECTER2's citation-graph-trained similarity). This is not a state-of-the-art or publication-quality evaluation — it's a single, honestly-reported run on the public validation split, with no reranking or hyperparameter tuning attempted.

### What is contaminated / not usable

The internal benchmark's generation-quality judge pass (faithfulness/relevance/completeness) reports a flat ~0.50 across all three metrics for every strategy — this is a known artifact of a Groq daily-quota exhaustion mid-run, not a real measurement, and is not cited as a result anywhere in this repository. A handful of older, differently-scaled benchmark runs (e.g. "BM25 baseline", "FAISS only") also exist in the live Benchmark UI's stored history with no corresponding code path in the current pipeline; they are not reproducible and are not reported here.

## Testing

```bash
pytest tests/
```

44 tests currently pass locally, with this breakdown:

| File | Tests | What it covers |
|---|---|---|
| `test_chunker_facts.py` | 2 | Short scientific-fact preservation during chunking |
| `test_metadata_resolver.py` | 3 | Metadata/year extraction logic |
| `test_papers_upload_security.py` | 17 | Path-traversal-safe filename handling and upload confinement |
| `test_retrieval_alignment.py` | 8 | Chunk/embedding contract alignment (mocked embedder) |
| `test_utils.py` | 13 | Checkpointing, math-preservation, chunk-filtering utilities |
| `test_collections_add_paper.py` | 1 | Live integration test: collection create → add paper → persist → remove |

CI (`.github/workflows/tests.yml`) runs **35 of the 44** — every exclusion is individually justified, not silently dropped:
- `test_collections_add_paper.py` (1 test) requires a live, corpus-seeded PostgreSQL instance, which CI does not provision. Run and passing locally against the project's own `docker compose` Postgres.
- `test_retrieval_alignment.py` (8 tests) mocks the actual embedding call, but importing `src/retrieval/embedder.py` still pulls in the real `sentence-transformers`/`torch` stack at module load time — installing that in CI just to satisfy an import, for tests whose logic never touches the model, was judged not worth the added CI runtime. Run and passing locally.

There is no code-coverage tooling configured in this repository, so no coverage percentage is claimed.

## Local setup / Docker

### Prerequisites
- Docker + Docker Compose
- (For local, non-Docker development) Python 3.11+, Node.js, and a local PostgreSQL 16 with the `pgvector` extension

### Docker (recommended)

```bash
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology
cp .env.example .env   # fill in the values you need — see below
docker compose up -d
```

This starts 5 containers: `api` (FastAPI, `:8000`), `db` (Postgres 16 + pgvector, `:5432`), `redis` (`:6379`), `ollama` (`:11434`), and `frontend` (React, served on `:5173`). The API image excludes secrets, local datasets, and internal docs from its build context via `.dockerignore`.

On a genuinely fresh database, the API's own startup (`api/main.py`'s lifespan hook) creates the current table schema automatically. If you're running against a pre-existing database that predates a migration (as happens when pulling updates), apply migrations explicitly:

```bash
docker compose exec api alembic upgrade head
```

**Health check**: `curl http://localhost:8000/health` → `{"status":"ok",...}`
**API**: `http://localhost:8000` (interactive docs at `/docs`)
**Frontend**: `http://localhost:5173`

### Without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn api.main:app --reload
```

```bash
cd frontend
npm install
npm run dev   # Vite will print the local dev URL
```

### Tests

```bash
pytest tests/
```

## Environment variables

Copy `.env.example` to `.env`. **Never commit real values** — the file below uses placeholders only.

| Variable | Required? | Purpose |
|---|---|---|
| `DATABASE_URL` | **Required** | Postgres connection string (`postgresql+asyncpg://user:pass@host:5432/dbname`) |
| `REDIS_URL` | **Required** | Redis connection string for response caching |
| `USE_GROQ` | Optional (default `false`) | `true` = use Groq for generation; `false` = use Ollama. A config switch, not automatic failover. |
| `GROQ_API_KEY` | **Provider-specific** — required only if `USE_GROQ=true` | Groq API key |
| `GROQ_MODEL` | Optional | Overrides the default Groq chat model |
| `COHERE_API_KEY` | **Required for reranking** | Cohere API key for `rerank-v3.5`; without it, retrieval falls back to unranked RRF-fused ordering |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Optional | Request tracing/observability; the app runs without them, just without traces |
| `OLLAMA_URL` | Optional (default `http://localhost:11434`) | Ollama server address, used when `USE_GROQ=false` |
| `EMBEDDING_MODEL` | Optional | Overrides the default embedding model (`allenai/specter2_base`) |
| `COHERE_RERANK_MODEL` | Optional | Overrides the default rerank model (`rerank-v3.5`) |

No Groq or Cohere key is strictly required to run the app read-only against an already-ingested corpus with `USE_GROQ=false` — generation runs against local Ollama and reranking falls back to unranked fusion.

## Deployment status

**Anthology is currently configured and verified for local Docker-based deployment only.** There is no public hosted instance and no live demo link — running it means cloning the repo and starting it yourself as described above. Deploying it publicly would additionally require, at minimum, authentication, rate limiting, and moving off trial-tier third-party API keys (see [Security](#security) and [Limitations](#limitations)).

## Limitations

- **No automatic generation-provider failover.** `USE_GROQ` is a static switch; a Groq failure at request time surfaces as an error, it does not retry Ollama.
- **Grounding is lexical, not semantic** — a word-overlap heuristic, not an entailment or hallucination-detection model.
- **The internal 247-question benchmark is self-generated and self-referential**, not an independent benchmark. The QASPER evaluation is the external check, and its results should not be conflated with the internal ones.
- **Generation-quality judge scores (faithfulness/relevance/completeness) for the internal benchmark are contaminated** by a Groq quota exhaustion mid-run and are not usable as evidence of anything.
- **Metadata extraction is heuristic-based**: title/author extraction is near-complete, but `arxiv_id`, `doi`, `abstract`, and `year` are missing for a meaningful fraction of ingested papers. This does not affect retrieval quality, only metadata completeness in the UI.
- **No authentication, no rate limiting.** Every route is reachable by anyone who can access the API port. Fine for local use; not fine for public exposure without further work.
- **No ANN/GIN indexing yet.** Both the vector similarity search and the full-text search run without an index tuned for scale (no `ivfflat`/`HNSW` index on embeddings, no GIN index on the FTS column). Acceptable at the current ~12K-chunk corpus size; would need addressing well before this reaches the hundreds of thousands of chunks.
- **Cohere reranking runs on a trial-tier key** (rate-limited to 10 requests/minute per the account tier); on repeated 429 responses the code falls back to unranked RRF-fused ordering after 3 retries.
- **Ollama runs locally and depends on local machine resources** — generation quality/speed with `USE_GROQ=false` is bounded by whatever hardware is running the Ollama container, not a managed cloud service.
- **A real bug was found and fixed during a hygiene pass**: the Collections feature's "add paper" endpoint previously returned a false-positive success while silently failing to persist, due to a missing database-level default on a timestamp column. Fixed with a migration, a code fix, and a regression test — mentioned here because it's the kind of thing worth being upfront about rather than omitting.
- **Not benchmarked at scale.** All numbers above describe a ~121-paper, ~12K-chunk corpus. Behavior at 10x or 100x that size is untested.

## Security

What's actually in place, verified against the code:
- **CORS is restricted**, not wildcarded — `allowed_origins` is `["http://localhost:3000", "http://localhost:5173"]` (`api/core/config.py`), not `["*"]`.
- **Path traversal protection on upload**: `_safe_pdf_filename()` reduces any uploaded filename to a safe basename (handling `../`, absolute paths, and Windows-style separators), plus a defense-in-depth check that the resolved destination path is actually contained within the upload directory before writing (`api/routers/papers.py`).
- **SQL injection protection**: all query construction uses SQLAlchemy's `text()` with bound parameters (`:vec`, `:k`, `:ct`, `:pid`, etc.) — never raw string interpolation of user input into SQL. The only string-templated part of any query is a small, hardcoded set of `WHERE`-clause fragments, never user-supplied values.
- **Secrets are read from environment variables** (`.env`, gitignored), not hardcoded in source.

What's explicitly **not** in place:
- **No authentication or authorization** on any route.
- **No rate limiting** on any route, including the upload and benchmark-trigger endpoints.
- **No comprehensive security audit** has been performed on this codebase — the items above are specific, verified protections against specific known issues, not a claim of general security hardening.

## Project structure

```
anthology/
├── api/                       # FastAPI application
│   ├── routers/                # papers, query, search, discovery, collections,
│   │                            # benchmark, sessions, feedback, stats, flowchart, suggest, health
│   ├── services/                # rag_service, paper_service, ingest_service
│   ├── models/                  # SQLAlchemy ORM tables
│   ├── schemas/                  # Pydantic request/response models
│   └── core/                    # config, database engine, model-name registry
├── src/
│   ├── ingestion/                # Docling parser, chunker, metadata_resolver
│   ├── retrieval/                # embedder (SPECTER2), retriever (hybrid pipeline)
│   ├── generation/                # generator (Groq/Ollama, streaming + non-streaming)
│   └── evaluation/                # evaluator, benchmarker, pipeline_runner
├── scripts/
│   ├── run_benchmark.py            # internal 6-strategy benchmark runner
│   └── evaluate_qasper.py          # external QASPER retrieval-generalization eval
├── benchmarks/
│   └── qa_dataset_v1.json           # the 247-question internal QA dataset
├── alembic/                    # database migrations
├── frontend/                   # React + TypeScript + Vite app
├── tests/                      # pytest suite (see Testing)
├── .github/workflows/           # CI
├── docker-compose.yml
├── Dockerfile / frontend/Dockerfile
└── .dockerignore / .env.example
```

## Technology stack

| Layer | Technology | Purpose |
|---|---|---|
| API framework | FastAPI | HTTP routing, request/response validation |
| Database | PostgreSQL 16 + `pgvector` | Paper/chunk storage, vector similarity search, full-text search |
| ORM | SQLAlchemy (async) + Alembic | Data access and schema migrations |
| Cache | Redis | Response caching |
| Embedding model | SPECTER2 (`allenai/specter2_base`, via `sentence-transformers`) | Scientific-paper-tuned dense embeddings |
| PDF parsing | Docling | Layout-aware parsing with OCR support |
| Reranking | Cohere `rerank-v3.5` | Cross-attention reranking of fused candidates |
| Generation (cloud) | Groq | LLM inference when `USE_GROQ=true` |
| Generation (local) | Ollama (`qwen2.5:7b` default) | LLM inference when `USE_GROQ=false` |
| Tracing | Langfuse | Request/generation observability |
| Frontend | React + TypeScript + Vite | UI |
| Containerization | Docker Compose | 5-service local orchestration |
| CI | GitHub Actions | Automated test run on push/PR |
| Testing | pytest, pytest-asyncio | Unit + integration tests |
| External data (discovery) | ArXiv API, OpenAlex API | Live paper discovery feature |
| External data (evaluation) | HuggingFace `datasets` (QASPER) | External benchmark corpus |

## Reproducing everything in this README

```bash
# 1. Clone and configure
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology
cp .env.example .env   # fill in values as needed (see Environment variables)

# 2. Start the stack
docker compose up -d
curl http://localhost:8000/health

# 3. Run the test suite
pytest tests/

# 4. Run the internal retrieval benchmark (writes indexes/results_*.json)
python scripts/run_benchmark.py

# 5. Run the external QASPER evaluation
#    (downloads QASPER from HuggingFace on first run; bounded pilot by default)
python scripts/evaluate_qasper.py --num-papers 40
python scripts/evaluate_qasper.py --num-papers 281 --split validation   # full validation split
```

Every command above was run against this repository before being included here.

## Why this project is technically interesting

- A genuinely **hybrid retrieval pipeline** — dense embeddings, sparse full-text search, RRF fusion, and cross-attention reranking — implemented, not just described, with each stage individually swappable for benchmarking.
- **Two independent evaluation tracks**: an internal strategy comparison and an external generalization check against a real third-party dataset (QASPER), with the external results reported honestly even where they contradict the internal ranking — the kind of finding a benchmark script optimized purely for a good-looking README would have buried.
- **Streaming RAG** with citation grounding built structurally from the chunks used in the prompt, over PostgreSQL/`pgvector` rather than a dedicated vector database.
- **Reproducibility as a design constraint**: every benchmark and test command in this README was actually run against this codebase before being written down, and the CI workflow's dependency list was verified in an isolated clean virtual environment, not just assumed to work from reading the code.
- A CI pipeline with **honest exclusions**: 35 of 44 tests run automatically, and the other 9 are individually justified (a live-database integration test and a heavy-ML-import test), not silently dropped from the count.

## License

MIT — see [LICENSE](LICENSE).
