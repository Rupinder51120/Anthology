# Anthology Repository Audit

Auditor: Principal Software Architect / Staff Engineer / Security Auditor / SRE  
Date: 2026-06-16  
Scope: first-party source, hidden/config files, manifests/locks, migrations, scripts, Docker/Compose, frontend, generated graph/index metadata, checked-in runtime artifacts, local data artifacts, and vendored/generated directory inventory.

Verdict: **NOT PRODUCTION READY**. The core RAG path is recognizable and partially functional, but production deployment is blocked by a broken Dockerfile, no passing frontend build, weak/no auth, fragile migrations, silent startup degradation, and operationally expensive local model assumptions.

## Audit Method

I enumerated files with `rg --files -uu` and `find`, including hidden files. The repository contains first-party code plus large generated/vendor/runtime directories: `.venv` (~1.3G), `frontend/node_modules` (~196M), `data` (~922M), `indexes` (~69M), and `graphify-out` (~2M). I reviewed first-party source, manifests, migrations, scripts, docs, generated metadata summaries, and artifact inventory. Vendored dependencies and binary/PDF/PNG assets were treated as supply-chain, storage, and repository-hygiene inputs, not line-by-line code.

Verification commands:

- `pytest -q`: **failed** because `pytest` is not on PATH in the active shell.
- `npm run build` in `frontend`: **failed** with TypeScript errors.
- `find . -maxdepth 2 -name requirements-cloud.txt -o -name start.sh`: returned no files, confirming Dockerfile missing inputs.
- `git status --short`: clean before audit rewrite.

## Phase 1: Project Understanding

Anthology is an AI research assistant for a local corpus of academic papers. It ingests PDFs, parses text/figures/tables, chunks content, embeds chunks with SPECTER2, stores chunks in PostgreSQL/pgvector, retrieves relevant chunks with pgvector plus PostgreSQL full-text search, optionally reranks with Cohere, and generates grounded answers using Groq or local Ollama. The active user-facing app is a React/Vite frontend backed by FastAPI. There is also a large legacy Streamlit app.

Likely users:

- Students/researchers asking questions across a local paper corpus.
- A maintainer ingesting PDFs and rebuilding indexes.
- A developer running offline retrieval/generation evaluation.

Primary workflows:

- Upload PDF -> parse -> chunk -> embed -> insert chunks/paper row.
- Ask question -> retrieve chunks -> generate answer -> return citations -> persist query.
- Browse/search papers.
- Discover/recommend papers from local index and external APIs.
- Run offline benchmark/evaluation scripts.

Core features:

- FastAPI backend, React frontend, PostgreSQL pgvector storage.
- RAG query endpoint and streaming query endpoint.
- PDF upload and ingestion pipeline.
- Local paper library, search, recommendations, flowchart, TTS, feedback, stats.
- Offline indexing and benchmark scripts.

Maturity level: **prototype to early beta**. There is real domain logic and some recent fixes, but the system still relies on local state, local models, hidden runtime assumptions, weak operational controls, and no test gate.

## Phase 2: Repository Map

Repository tree, abbreviated to meaningful ownership boundaries:

```text
anthology/
  api/
    main.py                 FastAPI app, router registration, startup model preload.
    core/
      config.py             Pydantic settings and env defaults.
      database.py           SQLAlchemy async engine/session and create_tables.
    models/tables.py        ORM models for Paper, Query, Feedback, Chunk.
    schemas/schemas.py      Pydantic API schemas.
    routers/                FastAPI endpoints.
    services/               RAG, ingestion, paper, stats, vector services.
  src/
    ingestion/              PDF parsing, metadata, chunking, figure/table helpers.
    retrieval/              Embedding, pgvector/FTS retrieval, HyDE.
    generation/             Groq/Ollama answer generation and conversation memory.
    discovery/              ArXiv/Semantic Scholar search.
    ui/                     Legacy/non-React helpers: recommend, TTS, flowchart.
    evaluation/             Offline benchmark and metric code.
    download/               ArXiv downloader.
  frontend/
    src/                    React/Vite app.
    package.json            Frontend scripts/deps.
    package-lock.json       Frontend lock.
  scripts/                  Offline indexing, migration, benchmark scripts.
  alembic/                  Database migrations.
  data/                     PDFs, uploads, figures, registry/config/report.
  indexes/                  FAISS/BM25/embeddings/metadata/eval outputs.
  graphify-out/             Generated code graph/cache/report.
  Dockerfile                Container build, currently broken.
  docker-compose.yml        Local API/db/redis stack, currently blocked by Dockerfile.
  .env.example              Incomplete env template.
  .gitignore                Ignores many generated dirs, but local artifacts still exist.
  README.md                 Product/dev docs.
  app.py                    Legacy Streamlit app, 2202 lines.
```

Important file status:

| File | Purpose | Status |
|---|---|---|
| `api/main.py` | Runtime entrypoint | Reachable; swallows startup failures. |
| `api/services/rag_service.py` | Main non-streaming RAG orchestration | Reachable; core path but has cache/session/observability issues. |
| `src/retrieval/retriever.py` | pgvector + FTS + RRF + Cohere | Reachable; filters by `source` for `paper_id`, likely wrong. |
| `src/generation/generator.py` | LLM prompt/call logic | Reachable; fallback to local Ollama not production-safe. |
| `api/routers/query.py` | Query endpoints | Reachable; streaming bypasses `RAGService` persistence/cache/memory/citations. |
| `api/routers/papers.py` | Paper/upload/vector endpoints | Reachable; upload filename/path handling is unsafe. |
| `api/core/database.py` | DB sessions | Reachable; dependency commits while services also commit. |
| `alembic/versions/*.py` | Migrations | Reachable operationally; schema history is inconsistent. |
| `frontend/src/*` | Main UI | Reachable; build fails. |
| `app.py` | Streamlit app | Potentially reachable if run manually; legacy/god file. |
| `Dockerfile` | Deployment | Broken. |
| `.env` | Local env | Present locally with secret variables; not tracked, but must be rotated if shared. |

## Phase 3: Architecture Truth Report

What developers likely think the architecture is:

```text
React frontend -> FastAPI routers -> services -> pgvector/Postgres + Redis + LLM providers
offline scripts -> ingestion/retrieval/evaluation modules
```

What it actually is:

```text
React frontend
  -> FastAPI route handlers that sometimes call services and sometimes inline business logic
  -> global in-process state for sessions/model caches
  -> Postgres used both through ORM and raw SQL
  -> Redis created per request
  -> external LLM APIs and local Ollama mixed behind env flags

Legacy Streamlit app
  -> directly imports retrieval/generation/download/ui modules
  -> has its own UI flows and assumptions

Offline scripts
  -> use direct asyncpg/SQLAlchemy and monkeypatch retriever behavior
```

Boundary violations:

- **VERIFIED**: Streaming query endpoint duplicates RAG orchestration instead of using `RAGService`. Evidence: `api/routers/query.py:31-39` retrieves/builds messages directly; `api/routers/query.py:41-58` calls Groq directly.
- **VERIFIED**: Services and routers own commits despite `get_db()` already committing. Evidence: dependency commits at `api/core/database.py:35-44`; `api/routers/feedback.py:38` commits; `api/services/ingest_service.py:105` commits; `api/services/paper_service.py:63` commits.
- **VERIFIED**: Business logic lives in routers (`api/routers/papers.py:64-91` raw vector/text search) and services (`api/services/rag_service.py`) and scripts (`scripts/build_index.py`).
- **VERIFIED**: State lives in globals: model cache in `src/retrieval/embedder.py:8-13`, `_sessions` and `_session_access` in `api/services/rag_service.py:27-28`, CrossEncoder cache in `src/retrieval/retriever.py:13`.
- **VERIFIED**: Side effects occur at import/startup and request time: startup loads DB/model in `api/main.py:13-23`; upload writes files in `api/routers/papers.py:28-33`; generation calls external services in `src/generation/generator.py:73-123`.

God files/services:

- **VERIFIED**: `app.py` is 2202 lines and mixes Streamlit UI, CSS, API calls, Mermaid rendering, chat, upload, search, benchmark display, and helpers.
- **VERIFIED**: `scripts/run_benchmark.py` is 601 lines and includes benchmark generation, monkeypatching, evaluation, cleanup, and CLI.
- **VERIFIED**: `src/generation/generator.py` is a god service for prompts, provider routing, streaming, grounding, citations, and response typing.
- **VERIFIED**: `src/ingestion/ingest.py` is 303 lines of PDF metadata extraction, loading, section extraction, and CLI behavior.

## Phase 4: Architecture Analysis

High-level architecture:

```mermaid
flowchart LR
  UI[React/Vite frontend] --> API[FastAPI]
  Legacy[Streamlit app.py] -. manual/legacy .-> Src[src modules]
  API --> Routers[routers]
  Routers --> Services[services]
  Services --> Retrieval[src/retrieval]
  Services --> Generation[src/generation]
  Retrieval --> PG[(Postgres + pgvector)]
  Retrieval --> Cohere[Cohere rerank]
  Generation --> Groq[Groq]
  Generation --> Ollama[Local Ollama fallback]
  Services --> Redis[(Redis optional cache)]
  Scripts[offline scripts] --> PG
  Scripts --> Indexes[indexes/*]
  Ingestion[src/ingestion] --> Data[data/*]
```

Request lifecycle for non-streaming query:

```mermaid
sequenceDiagram
  participant Browser
  participant FastAPI
  participant RAG
  participant Retriever
  participant DB
  participant LLM
  participant Redis

  Browser->>FastAPI: POST /api/v1/query
  FastAPI->>RAG: query(request, db)
  RAG->>Redis: get cache key
  Redis-->>RAG: cached or miss
  RAG->>Retriever: retrieve(question, top_k, db, paper_id)
  Retriever->>DB: pgvector search
  Retriever->>DB: FTS search
  Retriever-->>RAG: reranked chunks
  RAG->>LLM: generate_answer(chunks, history)
  LLM-->>RAG: answer
  RAG->>DB: add Query
  RAG->>Redis: set cache
  RAG-->>FastAPI: QueryResponse
  FastAPI-->>Browser: JSON
```

Database architecture:

- SQLAlchemy models define `papers`, `queries`, `feedback`, `chunks`.
- Migrations are incomplete/drifted relative to models.
- Runtime startup calls `Base.metadata.create_all`, which bypasses Alembic as the source of truth.
- Retrieval uses raw SQL for pgvector and FTS.

Authentication/authorization architecture:

- **VERIFIED**: None. No auth dependency is used by routers. `api/dependencies.py` only exposes settings dependency and is not imported by routers. All upload, query, feedback, discovery, benchmark, and stats endpoints are unauthenticated.

Event/background architecture:

- **VERIFIED**: No queue/worker/scheduler is defined in first-party code. Long PDF ingestion and model calls run inside request/script flows.

Deployment architecture:

- Docker Compose intends `api + db + redis`, but API image cannot currently build.

## Phase 5: Feature Inventory

| Feature | Entry points | Storage | Status | Evidence |
|---|---|---|---|---|
| Health | `GET /health` | chunks table | Partial | `api/routers/health.py:13-25`; only checks indexed chunks and hardcodes `ollama=False`. |
| Query | `POST /api/v1/query` | queries, chunks, Redis | Partial | `api/routers/query.py:12-18`, `api/services/rag_service.py:62-188`. |
| Streaming query | `POST /api/v1/query/stream` | chunks only | Partial/Broken behavior | `api/routers/query.py:21-68`; no citations, no persistence, Groq-only. |
| Paper list/detail | `GET /papers`, `GET /papers/{id}` | papers | Partial | `api/routers/papers.py:44-55`. |
| PDF upload/ingest | `POST /papers/upload` | data/papers, chunks, papers | Risky | `api/routers/papers.py:14-41`, `api/services/ingest_service.py`. |
| Vector/text search | `POST /vectors/search` | chunks | Partial | `api/routers/papers.py:64-91`; actually FTS, not vector. |
| Semantic search | `POST /search` | chunks | Partial | `api/routers/search.py:9-41`. |
| Recommendations | `POST /recommend` | indexes, ArXiv | Partial | depends on local index files and external ArXiv. |
| Discovery | `POST /discover` | external APIs | Partial | ArXiv/S2 unauthenticated outbound calls. |
| Flowchart | `POST /flowchart` | none | Experimental | local Ollama required. |
| TTS | `POST /tts` | temp files | Platform-specific | macOS `say` command only. |
| Feedback | `POST/GET /feedback` | feedback | Partial | double-commit risk. |
| Benchmark report | `GET /benchmark*` | indexes JSON | Static/Partial | returns files if present. |
| Legacy Streamlit UI | `streamlit run app.py` | API + local files | Legacy/High debt | `app.py` god file. |

## Phase 6: Dead Code and Artifact Candidates

| Item | Confidence | Evidence | Safe to delete? | Impact |
|---|---:|---|---|---|
| `multimodal/*` empty directories | VERIFIED | `find` reports empty `multimodal/ingestion`, `storage`, `retrieval`, `worker`. | Yes, if not planned. | Removes abandoned architecture signal. |
| `__pycache__` checked/local artifacts | VERIFIED | Many `.pyc` files found outside `.venv`. | Yes. | Reduces noise. |
| `frontend/src/pages/Stubs.tsx` duplicate upload page | LIKELY | Defines another `UploadPage`; `App.tsx` imports only `CollectionsPage`, `SettingsPage`. | Partial. | Keep stubs or split real pages. |
| `api/dependencies.py` | LIKELY | Only defines unused settings dependency. | Yes after search. | Low. |
| `api/services/vector_service.py` | LIKELY | No first-party import found except itself. | Maybe. | Could be future service for `/vectors/search`. |
| `src/ingestion/utils.py` | LIKELY | No first-party imports found. | Maybe. | Utility functions can be revived. |
| `src/evaluation/generation_metrics.py` | LIKELY | Not imported by scripts currently found. | No until evaluation scope decided. | Offline evaluation loss. |
| `src/evaluation/retrieval_metrics.py` | Potentially reachable | Imported by `scripts/run_benchmark.py`. | No. | Benchmark breaks. |
| `app.py` | Potentially reachable | Manual Streamlit entrypoint; not used by FastAPI/React. | No unless product drops Streamlit. | Removes legacy UI. |
| `graphify-out/cache/*` | VERIFIED generated | Generated graph cache. | Yes, regenerate. | Low. |
| `data/figures/*.png` | VERIFIED generated | Figure extraction outputs. | Maybe. | Needed for multimodal citations if image paths reference them. |
| `indexes/*.npy`, `*.bin`, `*.pkl` | VERIFIED generated | Vector/BM25/metadata indexes. | No unless rebuild pipeline is reliable. | App/recommend/eval can break. |
| `.venv/`, `frontend/node_modules/` | VERIFIED vendored local deps | Large local dirs; ignored by `.gitignore`. | Yes locally if reinstallable. | Frees ~1.5G. |

## Phase 7: Bug Analysis

### Critical / High

1. **VERIFIED - Docker build is broken**
   - Severity: Critical
   - Evidence: `Dockerfile:14-15` copies/installs `requirements-cloud.txt`; `Dockerfile:19-23` chmods/runs `start.sh`; neither file exists.
   - Repro: `docker compose up --build` will fail at copy or chmod.
   - Fix: use `requirements.txt` or create `requirements-cloud.txt`, add `start.sh`, or change `CMD` to `uvicorn api.main:app --host 0.0.0.0 --port 8000`.

2. **VERIFIED - Frontend cannot build**
   - Severity: Critical
   - Evidence: `npm run build` failed with TS6133 unused imports in `Chat.tsx`, `PaperView.tsx`, `Search.tsx`, `Stubs.tsx`, `Upload.tsx`, and TS2352 at `frontend/src/pages/Home.tsx:121`.
   - Repro: run `npm run build` in `frontend`.
   - Fix: remove unused imports and replace `(stats as Record<string, number>)[s.key]` with a typed key approach.

3. **VERIFIED - Startup hides fatal dependency failures**
   - Severity: High
   - Evidence: `api/main.py:13-23` wraps `create_tables()` and model preload in broad `except` and continues.
   - Repro: point `DATABASE_URL` at a dead DB; app may start and later fail requests.
   - Fix: fail startup for required dependencies; split liveness and readiness.

4. **VERIFIED - Alembic migrations do not create the embedding column before altering it**
   - Severity: High
   - Evidence: `alembic/versions/5890fefb391a...py:29-47` creates `chunks` but omits `embedding`; `specter2_migration.py:15` alters `chunks.embedding`.
   - Repro: run Alembic from empty DB instead of `create_tables`; `ALTER COLUMN embedding` can fail.
   - Fix: add `sa.Column('embedding', Vector(1024 or 768))` in the original migration or create an explicit add-column migration.

5. **VERIFIED - Runtime schema and migration schema drift**
   - Severity: High
   - Evidence: ORM `Chunk.chunk_id` is `String(32)` in `api/models/tables.py`, migration uses `String(20)` at `alembic/versions/5890...py:31`; ORM `Paper` has `figure_count`/`table_count`, migration lacks them; migration has `updated_at`, ORM lacks it.
   - Fix: make Alembic the source of truth, generate a reconciliation migration, stop relying on `create_all` in production.

6. **VERIFIED - Upload allows path traversal / arbitrary filename write under project-relative path**
   - Severity: High
   - Evidence: `api/routers/papers.py:22` checks only suffix; `api/routers/papers.py:28` uses `Path("data/papers") / file.filename`.
   - Repro: multipart filename like `../x.pdf` or path separators can escape intended naming rules depending server normalization.
   - Fix: use `Path(file.filename).name`, generate server-side UUID names, validate MIME/signature, store original name separately.

7. **VERIFIED - No authentication or authorization**
   - Severity: High
   - Evidence: routers use `Depends(get_db)` but no user/auth dependency; all APIs are registered in `api/main.py:45-55`.
   - Impact: anyone with network access can upload files, query private papers, view benchmark/status, and submit feedback.
   - Fix: add auth middleware/dependencies and authorization checks by corpus/user.

8. **LIKELY - `paper_id` filtering is semantically wrong**
   - Severity: High
   - Evidence: `QueryRequest.paper_id` is optional; retriever filters `source = :pid` in `src/retrieval/retriever.py:55-57` and `91-93`, while paper IDs are UUIDs and chunk `source` is filename.
   - Repro: PaperView sends UUID as `paper_id`; retrieval returns no chunks.
   - Fix: add `paper_id` FK to chunks and filter on it, or pass filename/source from frontend.

9. **VERIFIED - Streaming query bypasses citations, cache, query persistence, memory, Langfuse, fallback**
   - Severity: Medium/High
   - Evidence: `api/routers/query.py:31-58` implements its own retrieve/Groq stream.
   - Impact: streamed chat lacks DB records and citations; Groq errors are emitted as normal SSE data.
   - Fix: move streaming into `RAGService` and share retrieval/memory/persistence.

10. **VERIFIED - Double transaction ownership**
    - Severity: Medium
    - Evidence: `get_db` commits at `api/core/database.py:39`; feedback commits at `api/routers/feedback.py:38`; ingest commits at `api/services/ingest_service.py:105`; paper sync commits at `api/services/paper_service.py:63`.
    - Fix: one owner. Prefer dependency-managed commit for request scope; services flush only.

### Medium / Low

- **VERIFIED**: Redis connection is opened and closed per query in `api/services/rag_service.py:51-58`, adding overhead. Use app-level pool/client.
- **VERIFIED**: Cache key ignores `paper_id`, `session_id`, `use_hyde`, and `retrieval_mode` in `api/services/rag_service.py:46-49`, which can serve wrong answers across scopes.
- **VERIFIED**: Health reports `ollama=False` unconditionally in `api/routers/health.py:21-25`, so the field is not a real dependency check.
- **VERIFIED**: `frontend/src/api/client.ts:65-68` defines an EventSource GET for a POST-only streaming route; currently unused but misleading.
- **VERIFIED**: `frontend/src/api/client.ts:27-39` expects `upload_date`, while backend `PaperOut` exposes `created_at` in `api/schemas/schemas.py:40-52`.
- **VERIFIED**: `api/routers/papers.py:64-91` endpoint is named `/vectors/search` but performs FTS only.

## Phase 8: Security Audit

Security findings:

| Severity | Confidence | Finding | Evidence | Fix |
|---|---|---|---|---|
| Critical | VERIFIED | Local `.env` contains secret variable names and likely live values. | `.env` keys include `GROQ_API_KEY`, `COHERE_API_KEY`, `LANGFUSE_SECRET_KEY`. Values intentionally not reproduced. | Rotate keys if this workspace was shared; keep `.env` untracked; use secret manager. |
| High | VERIFIED | No auth/authorization on API. | All routers registered publicly in `api/main.py:45-55`; no auth dependencies. | Add authentication and corpus ownership. |
| High | VERIFIED | Upload filename/path traversal and untrusted PDF processing. | `api/routers/papers.py:22-33`. | Sanitize/generate filenames, content-type sniffing, AV/sandbox parsing, size and page count limits. |
| High | VERIFIED | CORS allows credentials. | `api/main.py:36-42` has `allow_credentials=True`. | Disable credentials unless cookie auth is implemented; restrict prod origins. |
| Medium | VERIFIED | Sensitive provider errors can be returned to users. | `api/routers/query.py:56-57`, `src/generation/generator.py:225-227`. | Log internal errors, return generic message with correlation ID. |
| Medium | VERIFIED | Benchmark/static report endpoints may expose internal eval data. | `api/routers/benchmark.py`. | Gate behind auth or disable in prod. |
| Medium | VERIFIED | External API calls use user-controlled query. | `src/discovery/*.py`. | Add rate limits, request budgets, allowlist destinations, abuse prevention. |
| Medium | VERIFIED | Prompt injection / data exfiltration risk in RAG. | Raw chunks inserted into prompt in `src/generation/generator.py:183-193`. | Add prompt-injection mitigations, source isolation, citations validation. |
| Medium | LIKELY | Dependency surface is very large. | `requirements.txt` includes cloud, notebook, test, Streamlit, FastAPI, LLM, vector, and heavy ML packages. | Split prod/dev/eval requirements; run `pip-audit`/Dependabot. |
| Low | VERIFIED | `window.open` without `rel` in Library. | `frontend/src/pages/Library.tsx:100`. | Use `noopener,noreferrer`. |

Secret rotation checklist:

- Rotate Groq, Cohere, Langfuse keys.
- Revoke any keys exposed in screenshots, attachments, local logs, or chat transcripts.
- Replace local `.env` with new values after rotation.
- Confirm `.env` is untracked with `git ls-files .env`.
- Add production secrets through platform secret manager.

Safe `.gitignore` additions:

- `*.tsbuildinfo`
- `frontend/dist/`
- `data/uploads/`
- `data/figures/`
- `*.pdf`
- `*.pkl`
- `*.npy`
- `*.bin`
- `*.pyc` and `__pycache__/` are already intended but local artifacts remain.

Production security checklist:

- AuthN/AuthZ, rate limits, request body limits, upload sandboxing.
- HTTPS-only, strict CORS, no credentials unless needed.
- Structured logs without secrets.
- Dependency scanning for Python and npm.
- Secret manager and key rotation procedure.
- RAG prompt-injection and data-boundary tests.

## Phase 9: Environment Audit

| Variable | Used | Required | Secret | Description |
|---|---|---|---|---|
| `DATABASE_URL` | Yes | Yes | Yes-ish | Postgres URL; contains credentials. |
| `REDIS_URL` | Yes | No | Sometimes | Optional cache URL. |
| `DEBUG` | Yes | No | No | Controls SQL echo. |
| `PYTHONPATH` | Yes | Dev/deploy | No | Import path. |
| `GROQ_API_KEY` | Yes | If `USE_GROQ=true` or streaming | Yes | Groq API key. |
| `USE_GROQ` | Yes | No | No | Switch Groq vs Ollama. |
| `GROQ_MODEL` | Yes | No | No | Text model name. |
| `COHERE_API_KEY` | Yes | No | Yes | Optional reranker key. |
| `LANGFUSE_PUBLIC_KEY` | Yes | No | No | Observability public key. |
| `LANGFUSE_SECRET_KEY` | Yes | No | Yes | Observability secret. |
| `LANGFUSE_HOST` | Yes | No | No | Langfuse host. |
| `OLLAMA_URL` | Partially | No | No | Used by ingestion helpers, not generator. |
| `TOKENIZERS_PARALLELISM` | Yes-ish | No | No | Tokenizer warning control. |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | Example only | No | No | HF warning control. |

`.env.example` is incomplete. A better template:

```env
DATABASE_URL=postgresql+asyncpg://anthology:anthology@localhost:5432/anthology
REDIS_URL=redis://localhost:6379
DEBUG=false
PYTHONPATH=.

USE_GROQ=false
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

COHERE_API_KEY=

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://us.cloud.langfuse.com

OLLAMA_URL=http://localhost:11434
TOKENIZERS_PARALLELISM=false
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

## Phase 10: API Analysis

| Method | Path | Auth | Request | Response | Status |
|---|---|---|---|---|---|
| GET | `/` | None | none | dict | OK |
| GET | `/health` | None | none | `HealthResponse` | Partial |
| POST | `/api/v1/query` | None | `QueryRequest` | `QueryResponse` | Partial |
| POST | `/api/v1/query/stream` | None | `QueryRequest` | SSE | Partial |
| GET | `/api/v1/papers` | None | none | `PaperListResponse` | OK |
| GET | `/api/v1/papers/{paper_id}` | None | UUID | `PaperOut` | OK |
| POST | `/api/v1/papers/upload` | None | multipart PDF | dict | Risky |
| POST | `/api/v1/papers/sync` | None | none | dict | Risky |
| POST | `/api/v1/vectors/search` | None | query params | dict | Misnamed |
| POST | `/api/v1/search` | None | `SearchRequest` | `SearchResponse` | Partial |
| POST | `/api/v1/recommend` | None | `RecommendRequest` | `RecommendResponse` | Partial |
| POST | `/api/v1/tts` | None | `TTSRequest` | WAV | Platform-specific |
| POST | `/api/v1/flowchart` | None | `FlowchartRequest` | `FlowchartResponse` | Experimental |
| GET | `/api/v1/benchmark` | None | none | JSON file | Static |
| GET | `/api/v1/benchmark/report` | None | none | JSON file | Static |
| POST | `/api/v1/feedback` | None | `FeedbackRequest` | `FeedbackResponse` | Partial |
| GET | `/api/v1/feedback/{query_id}` | None | path string | dict | Type mismatch risk |
| GET | `/api/v1/stats` | None | none | `StatsResponse` | OK |
| POST | `/api/v1/discover` | None | `DiscoveryRequest` | dict | Partial |

Undocumented/broken/misaligned:

- `/api/v1/vectors/search` is FTS, not vector.
- Streaming endpoint sends no structured error event and no citations.
- All endpoints are unauthenticated.
- Frontend build failure prevents reliable API consumption.

## Phase 11: Database Review

Schema:

- `papers`: paper metadata, unique filename/arxiv.
- `chunks`: chunk metadata, text, multimodal fields, vector embedding.
- `queries`: persisted Q/A.
- `feedback`: rating/comment per query.

Issues:

- **VERIFIED**: Migration drift, as described above.
- **VERIFIED**: Runtime `create_all` in `api/core/database.py:47-49` can mask missing migrations.
- **VERIFIED**: No explicit index for FTS expression despite repeated `to_tsvector('english', text)` queries.
- **LIKELY**: Missing `paper_id` relationship on chunks forces filename/source coupling and breaks UUID filtering.
- **LIKELY**: No cascade behavior on feedback/query deletion.
- **LIKELY**: JSON citations are untyped; long-term analytics will be harder.

Recommended fixes:

1. Stop `create_all` in production; run Alembic migrations.
2. Reconcile migrations with ORM models.
3. Add `chunks.paper_id` FK and backfill.
4. Add GIN index on `to_tsvector('english', text)`.
5. Add HNSW/IVFFlat pgvector index in migrations, not ad hoc scripts.

## Phase 12: Frontend Review

Architecture:

- React 19, Vite, TypeScript, TanStack Query, Zustand.
- Routes in `frontend/src/App.tsx`.
- API wrapper in `frontend/src/api/client.ts`.
- Heavy inline styles and shared CSS variables.

Findings:

- **VERIFIED**: Build fails under `tsconfig.app.json:19-20` strict unused checks and type check at `Home.tsx:121`.
- **VERIFIED**: Streaming client ignores HTTP non-OK and assumes `res.body!` at `frontend/src/api/client.ts:77-83`.
- **VERIFIED**: UI hardcodes "122 papers indexed" in `frontend/src/pages/Chat.tsx:169` and `:428`.
- **VERIFIED**: API `Paper` type expects `upload_date`, backend returns `created_at`.
- **LIKELY**: Search input on home does not pass the typed query to `/search`; it only navigates.
- **LIKELY**: Accessibility gaps from many custom buttons without labels/tooltips and inline focus handling.

## Phase 13: Backend Review

Backend strengths:

- Clear FastAPI router separation.
- Pydantic validation for main inputs.
- SQL queries use bound parameters in current retrieval/search paths.
- Recent fixes improved DB pool and search route behavior.

Backend risks:

- No auth/rate limiting.
- Broad exception swallowing.
- Request-time long tasks for upload and generation.
- Mixed ORM and raw SQL.
- Inconsistent transaction ownership.
- Env access split between settings and direct `os.getenv`.
- Global mutable state for sessions/models.

## Phase 14: Performance Review

Main bottlenecks:

- **VERIFIED**: SPECTER2 model loads at startup in `api/main.py:18-19`; expensive cold start and memory footprint.
- **VERIFIED**: Embedding query runs in executor per request in `src/retrieval/retriever.py:199-205`.
- **VERIFIED**: Full-text search computes `to_tsvector` at query time in `src/retrieval/retriever.py:83-109` and `api/routers/papers.py:71-79`.
- **VERIFIED**: Redis client/ping per query in `api/services/rag_service.py:51-58`.
- **VERIFIED**: Upload reads whole file into memory at `api/routers/papers.py:31`.
- **LIKELY**: Large index/data files in repo increase clone/build/deploy time.

Optimizations:

- App-level Redis pool.
- Streaming file writes.
- Background ingestion queue.
- Generated FTS column or expression GIN index.
- Model warmup/readiness separate from liveness.
- Cache key includes scope and retrieval mode.

## Phase 15: Cost Analysis

Cost drivers:

- Groq generation and vision model calls.
- Cohere rerank per query.
- Langfuse traces.
- CPU/RAM for SPECTER2 and optional CrossEncoder.
- Storage for PDFs, figures, embeddings, generated metadata.
- Postgres vector index/storage.

Waste:

- `requirements.txt` pulls cloud/notebook/test/Streamlit dependencies into likely API image.
- Vendored local `.venv` and `node_modules` occupy ~1.5G locally.
- Streaming and non-streaming query duplicate paths and may duplicate provider calls during debugging.

Savings:

- Split prod/dev/eval requirements.
- Cache embeddings and rerank results carefully by scoped key.
- Make Cohere optional per route/user tier.
- Move artifacts to object storage or ignored local cache.

## Phase 16: AI/LLM Audit

RAG implementation:

- Embeddings: SPECTER2 (`allenai/specter2_base`) with 768-dim vectors.
- Retrieval: pgvector search + PostgreSQL FTS + RRF.
- Reranking: optional Cohere.
- Generation: Groq or Ollama.
- Context: max 4000 chars in `src/generation/generator.py:126-141`.
- Memory: in-process per-session `ConversationMemory`.

Risks:

- **VERIFIED**: Prompt claims "Cover EVERY source" but context truncation can silently omit chunks.
- **VERIFIED**: `_is_grounded` is a lexical overlap heuristic in `src/generation/generator.py:171-180`, not factual grounding.
- **VERIFIED**: Streaming endpoint does not use same fallback/provider routing as `generate_answer`.
- **LIKELY**: Prompt injection in papers can affect answers.
- **LIKELY**: Token/cost tracking is incomplete for Groq (`tokens_used = 0`).

Recommendations:

- Centralize prompt templates and version them.
- Add source-span citation validation.
- Add eval set for hallucination/refusal/insufficient-context.
- Track provider token usage.
- Add RAGAS or similar evaluation as CI/nightly, not request path.

## Phase 17: Testing Review

Current evidence:

- No `tests/` directory was found in first-party file listing.
- `pytest -q` could not run because `pytest` is not on PATH in the active environment, despite being listed in `requirements.txt`.
- Frontend build fails.

Missing tests:

- API route tests with test DB.
- Retrieval unit tests and integration tests with seeded chunks.
- Upload security tests for filename/path traversal and malformed PDFs.
- Migration test from empty DB.
- Frontend build/type check in CI.
- RAG golden-answer tests.
- Auth tests after auth is added.

## Phase 18: CI/CD and Deployment Review

Findings:

- **VERIFIED**: No `.github/workflows` found in file inventory.
- **VERIFIED**: Dockerfile is broken.
- **VERIFIED**: Compose depends on API build, so compose is broken until Dockerfile is fixed.
- **VERIFIED**: No Kubernetes/Terraform found.
- **VERIFIED**: No production start command exists because `start.sh` is missing.

Ideal deployment:

- Backend image built from slim prod requirements.
- Managed Postgres with pgvector.
- Managed Redis.
- Object storage for PDFs/figures.
- Background worker for ingestion.
- CI: Python lint/type/test, Alembic migration test, frontend lint/build, dependency scan, Docker build.

## Phase 19: Production Readiness Review

What breaks first:

- Docker build fails immediately.
- If bypassed, frontend build fails.
- If run locally, startup can hide DB/model failures.
- Under load, upload/query latency and memory pressure rise.
- During restart, in-memory sessions vanish.
- During third-party outages, Groq/Cohere/Langfuse failures degrade inconsistently.

Failure scenarios:

| Scenario | Impact | Likelihood | Fix |
|---|---|---:|---|
| Fresh Docker deploy | No API image | High | Fix Dockerfile/start command. |
| Frontend CI build | No deployable frontend | High | Fix TS errors. |
| DB migration from empty DB | Migration failure/drift | High | Reconcile Alembic. |
| User uploads malicious filename | File write outside intended path | Medium/High | Sanitize/generate filenames. |
| Groq key missing and no Ollama | Query generation fails | High in cloud | Make provider config explicit and readiness-checked. |
| Redis down | Cache disabled silently | Medium | Log and expose degraded dependency state. |
| Large PDF upload | Memory spike/request timeout | Medium | Stream upload and async ingestion. |

## Phase 20: Code Quality Review

| Priority | Refactor | Effort | Impact |
|---|---|---:|---|
| P0 | Fix Dockerfile/start command | 1 hour | Enables deploy. |
| P0 | Fix frontend TS build | 1-2 hours | Enables frontend CI/deploy. |
| P0 | Reconcile migrations | 1 day | Prevents DB bootstrap failure. |
| P0 | Secure upload filenames | 1-2 hours | Closes high security risk. |
| P1 | Add auth/rate limits | 2-5 days | Makes API safe to expose. |
| P1 | Move streaming into `RAGService` | 1-2 days | Removes behavior drift. |
| P1 | Background ingestion worker | 3-5 days | Improves reliability. |
| P1 | Split prod/dev/eval dependencies | 1 day | Smaller images and less risk. |
| P2 | Replace `app.py` or clearly mark legacy | 2-5 days | Reduces confusion. |
| P2 | Add typed repositories/services | 1-2 weeks | Cleaner boundaries. |

## Phase 21: Improvement Roadmap

Phase 1: safe improvements, 1-2 days

- Fix Dockerfile and add `start.sh` or direct `uvicorn` command.
- Fix frontend build errors.
- Update `.env.example`.
- Sanitize upload filenames.
- Add basic Makefile/task docs for backend/frontend.
- Add health/readiness distinction.

Phase 2: medium-risk improvements, 1-2 weeks

- Reconcile Alembic migrations and remove prod `create_all`.
- Add CI pipeline.
- Add auth/rate limiting.
- Add tests for query/search/upload/migrations.
- Centralize settings instead of scattered `os.getenv`.
- Fix streaming/non-streaming parity.

Phase 3: major refactors, 1-2 months

- Add ingestion worker/queue.
- Add object storage abstraction.
- Add `papers`/`chunks` FK and repository layer.
- Version prompts and model configs.
- Decompose legacy Streamlit or remove it.

Phase 4: architecture modernization

- Multi-tenant corpus model with authorization.
- Observability: traces, metrics, structured logs.
- Evaluation pipeline as scheduled job.
- Artifact registry for indexes.
- Production RAG guardrails and cost controls.

## Phase 22: Documentation Generation Plan

The repository should have these docs generated or rewritten:

- `README.md`: quickstart, architecture overview, dev commands, env setup, known limitations.
- `docs/architecture.md`: diagrams from this audit, request/data flows, service boundaries.
- `docs/api.md`: endpoint table, schemas, auth, examples.
- `docs/security.md`: threat model, upload handling, secrets, auth, RAG risks.
- `docs/deployment.md`: Docker/Compose/cloud deploy, migrations, readiness, rollback.
- `docs/developer-onboarding.md`: read order, local setup, test strategy.
- `docs/troubleshooting.md`: DB, pgvector, model, Groq/Ollama, Redis failures.
- `docs/decision-log.md`: why pgvector, why SPECTER2, why Groq/Ollama, future worker.

Immediate doc warning to add: "This app is not safe to expose publicly until auth, upload hardening, and deployment fixes are complete."

## Phase 23: Knowledge Transfer Report

Most confusing things for a new team:

1. FastAPI/React is active, but `app.py` is a huge legacy Streamlit app.
2. Migrations and ORM disagree.
3. Some indexes are source of truth for features, others are generated outputs.
4. Streaming and non-streaming query behave differently.
5. `paper_id` appears to be a UUID but retrieval filters by filename/source.
6. Local Ollama is assumed by many helpers.
7. `.env.example` omits important variables.
8. Docker does not match repo files.

10 most important files:

1. `api/main.py`
2. `api/core/config.py`
3. `api/core/database.py`
4. `api/models/tables.py`
5. `api/services/rag_service.py`
6. `src/retrieval/retriever.py`
7. `src/generation/generator.py`
8. `api/services/ingest_service.py`
9. `frontend/src/api/client.ts`
10. `Dockerfile`

10 most dangerous files to modify:

1. `api/models/tables.py`
2. `alembic/versions/*.py`
3. `src/retrieval/retriever.py`
4. `src/retrieval/embedder.py`
5. `src/generation/generator.py`
6. `api/core/database.py`
7. `scripts/build_index.py`
8. `indexes/chunks_metadata.json`
9. `data/download_registry.json`
10. `.env`

Read first:

1. `README.md`
2. `api/main.py`
3. `api/routers/query.py`
4. `api/services/rag_service.py`
5. `src/retrieval/retriever.py`
6. `src/generation/generator.py`
7. `api/models/tables.py`
8. `alembic/versions/*.py`
9. `frontend/src/api/client.ts`
10. `Dockerfile`

## Phase 24: Final Verdict

1. This system is a local/hosted academic-paper RAG assistant.
2. The actual architecture is FastAPI + React plus legacy Streamlit, Postgres/pgvector, Redis optional cache, local/generated artifacts, and mixed cloud/local LLM calls.
3. The architecture is plausible for a prototype but not disciplined enough for production yet.
4. Dead code candidates include empty `multimodal/*`, pycache artifacts, generated graph caches, unused dependencies/service modules, and legacy Streamlit if React is the product.
5. Unfinished areas: auth, deployment, migrations, upload security, CI/tests, docs, background jobs.
6. Biggest bugs: Docker build failure, frontend build failure, migration drift, wrong `paper_id` filter, streaming/non-streaming drift.
7. Biggest security risks: no auth, unsafe upload filename handling, local secrets, broad exposed APIs, RAG prompt injection.
8. This cannot go to production today.
9. Fix first: Dockerfile, frontend build, migrations, upload sanitization, auth/rate limits.
10. First 30 days: stabilize deploy/test/migration/auth, then refactor RAG/ingestion boundaries, then add observability/evaluation/cost controls.

## Scores

| Area | Score |
|---|---:|
| Architecture | 55 |
| Code Quality | 52 |
| Security | 25 |
| Performance | 45 |
| Maintainability | 48 |
| Testing | 10 |
| Documentation | 45 |
| Scalability | 35 |
| Production Readiness | 20 |

## Top Lists

Top 20 issues:

1. Dockerfile references missing `requirements-cloud.txt`.
2. Dockerfile references missing `start.sh`.
3. Frontend build fails.
4. No API auth.
5. Unsafe upload filename handling.
6. Alembic migrations omit/alter missing embedding column.
7. ORM/migration schema drift.
8. Startup swallows fatal failures.
9. Streaming query bypasses service behavior.
10. Wrong/likely wrong paper ID filtering.
11. Double commit ownership.
12. Incomplete `.env.example`.
13. No CI pipeline.
14. No tests found/runnable.
15. No background ingestion worker.
16. Local Ollama assumptions in cloud.
17. FTS without evident expression index.
18. Huge local artifacts in workspace.
19. Legacy god-file Streamlit app.
20. Cache key ignores scope.

Top 20 bugs:

1. Docker build failure.
2. Frontend TS2352 at `Home.tsx:121`.
3. Unused imports causing TS6133 build failures.
4. Migration `specter2_768` alters absent embedding column.
5. `paper_id` UUID filtered against `chunks.source`.
6. Streaming endpoint no citations/persistence.
7. EventSource helper uses GET against POST route.
8. Upload reads entire file into memory.
9. Upload extension check is case-sensitive and path-unsafe.
10. Health `ollama` is always false.
11. Cache key omits paper/session/retrieval mode.
12. Query cache can return answer from different scope.
13. Services commit inside dependency-managed session.
14. Groq token usage recorded as 0.
15. Frontend `Paper` type mismatches backend `created_at`.
16. `/vectors/search` is not vector search.
17. Docker Compose API cannot build.
18. `create_all` can mask migration drift.
19. Benchmark endpoints assume local files.
20. Prompt grounding is lexical only.

Top 20 security risks:

1. No authentication.
2. No authorization.
3. Unsafe upload paths.
4. Untrusted PDF parsing in request path.
5. Local `.env` contains sensitive keys.
6. No rate limits.
7. CORS credentials enabled.
8. Public benchmark/internal endpoints.
9. Provider errors returned to clients.
10. RAG prompt injection.
11. No secret manager.
12. No dependency scanning.
13. Large attack surface in prod dependencies.
14. Discovery endpoints can be abused for outbound traffic.
15. No audit logs.
16. No content moderation/abuse controls.
17. No request size middleware beyond upload file size.
18. No per-user corpus boundaries.
19. No CSRF model documented if credentials later used.
20. Potential reverse-tabnabbing in Library `window.open`.

Top 20 improvements:

1. Fix Dockerfile.
2. Fix frontend build.
3. Add CI.
4. Reconcile migrations.
5. Add auth.
6. Sanitize uploads.
7. Add rate limits.
8. Add background ingestion.
9. Split dependencies.
10. Centralize settings.
11. Fix `paper_id` data model.
12. Unify query/streaming service.
13. Add tests.
14. Add readiness checks.
15. Add structured logging.
16. Add FTS/vector indexes in migrations.
17. Improve cache key.
18. Add prompt/version management.
19. Document architecture/API/deploy/security.
20. Remove/mark legacy and generated artifacts.

Top 20 files needing attention:

1. `Dockerfile`
2. `frontend/src/pages/Home.tsx`
3. `frontend/src/pages/Chat.tsx`
4. `api/main.py`
5. `api/routers/papers.py`
6. `api/routers/query.py`
7. `api/services/rag_service.py`
8. `src/retrieval/retriever.py`
9. `src/generation/generator.py`
10. `api/core/database.py`
11. `api/models/tables.py`
12. `alembic/versions/5890fefb391a_add_chunks_table_with_pgvector.py`
13. `alembic/versions/specter2_migration.py`
14. `.env.example`
15. `requirements.txt`
16. `frontend/src/api/client.ts`
17. `api/services/ingest_service.py`
18. `api/services/paper_service.py`
19. `scripts/build_index.py`
20. `app.py`

Top 20 dead code candidates:

1. `multimodal/ingestion`
2. `multimodal/storage`
3. `multimodal/retrieval`
4. `multimodal/worker`
5. `data/tables`
6. `__pycache__/`
7. `api/**/__pycache__/`
8. `src/**/__pycache__/`
9. `graphify-out/cache/`
10. `api/dependencies.py`
11. `api/services/vector_service.py`
12. `src/ingestion/utils.py`
13. duplicate upload code in `frontend/src/pages/Stubs.tsx`
14. unused `streamQuery` EventSource helper
15. local `.venv/`
16. local `frontend/node_modules/`
17. generated `indexes/*` if rebuildable
18. generated `data/figures/*` if rebuildable
19. legacy `app.py` if React is canonical
20. stale `anthology_audit.md` content replaced by this report

Top 20 highest-risk areas:

1. Public unauthenticated API.
2. File upload and PDF parsing.
3. Database migrations.
4. Docker deployment.
5. Frontend build.
6. Query retrieval correctness.
7. Streaming query divergence.
8. Secret handling.
9. LLM provider outage behavior.
10. Local model memory/cold start.
11. Postgres vector/FTS performance.
12. Artifact/data management.
13. Legacy Streamlit app.
14. No CI/tests.
15. Prompt injection.
16. Transaction ownership.
17. Redis cache correctness.
18. Background job absence.
19. Dependency bloat.
20. Documentation drift.
