# Anthology — Production Readiness Audit

> **Auditor**: Principal Software Engineer  
> **Date**: 2026-06-14  
> **Verdict**: **PARTIALLY WORKING** — Core query pipeline functions; upload pipeline has a critical runtime bug; multiple claimed features are broken or unreachable.

---

## Phase 1: Repository Map

```
anthology/
├── api/
│   ├── __init__.py                       (empty, package marker)
│   ├── main.py                           Purpose: FastAPI entrypoint
│   │                                     Imported by: uvicorn (runtime)
│   │                                     Imports: config, database, all 10 routers
│   │                                     Status: ✅ Working
│   │
│   ├── dependencies.py                   Purpose: FastAPI dependency helpers
│   │                                     Imported by: NOTHING — never used
│   │                                     Imports: get_db, get_settings
│   │                                     Status: ⚠️ Dead code — no router imports it
│   │
│   ├── core/
│   │   ├── __init__.py                   (empty)
│   │   ├── config.py                     Purpose: Pydantic Settings with lru_cache
│   │   │                                 Imported by: main, database, health, papers, paper_service, stats_service
│   │   │                                 Imports: pydantic_settings
│   │   │                                 Status: ⚠️ Partially working — missing `chunks_path` attribute (see Phase 4)
│   │   │
│   │   └── database.py                   Purpose: AsyncPG engine, session factory, Base
│   │                                     Imported by: main, health, papers, query, feedback, stats, alembic/env
│   │                                     Imports: config
│   │                                     Status: ✅ Working
│   │
│   ├── models/
│   │   ├── __init__.py                   (empty)
│   │   └── tables.py                     Purpose: SQLAlchemy ORM models (Paper, Query, Feedback, Chunk)
│   │                                     Imported by: feedback, paper_service, stats_service, alembic/env
│   │                                     Imports: database.Base, pgvector
│   │                                     Status: ✅ Working
│   │
│   ├── schemas/
│   │   ├── __init__.py                   (empty)
│   │   └── schemas.py                    Purpose: Pydantic request/response models
│   │                                     Imported by: all routers
│   │                                     Status: ✅ Working
│   │
│   ├── routers/
│   │   ├── __init__.py                   (empty)
│   │   ├── health.py                     Status: ✅ Working
│   │   ├── query.py                      Status: ✅ Working (non-streaming), ❌ Broken (streaming — see Phase 3)
│   │   ├── papers.py                     Status: ❌ Broken (upload) / ✅ Working (list/get/sync partial)
│   │   ├── search.py                     Status: ❌ Broken (calls old sync `retrieve` signature)
│   │   ├── recommend.py                  Status: ⚠️ Partially working (depends on pre-built index files)
│   │   ├── tts.py                        Status: ⚠️ macOS-only — broken on Linux/cloud
│   │   ├── flowchart.py                  Status: ⚠️ Requires local Ollama
│   │   ├── benchmark.py                  Status: ✅ Working (reads static JSON)
│   │   ├── feedback.py                   Status: ✅ Working
│   │   └── stats.py                      Status: ✅ Working
│   │
│   └── services/
│       ├── __init__.py                   (empty)
│       ├── rag_service.py                Status: ✅ Working — core query pipeline
│       ├── ingest_service.py             Status: ⚠️ Implemented but unreachable from upload route
│       ├── paper_service.py              Status: ❌ Broken — `settings.chunks_path` does not exist
│       ├── vector_service.py             Status: ⚠️ Dead code — never imported by any router or service
│       └── stats_service.py              Status: ⚠️ Hardcoded wrong embedding_dim (1024 vs 768)
│
├── src/
│   ├── __init__.py                       (empty)
│   ├── ingestion/
│   │   ├── __init__.py                   (empty)
│   │   ├── ingest.py                     Purpose: PDF loading, metadata, section extraction
│   │   │                                 Imported by: ingest_service, build_index, embedder(__main__)
│   │   │                                 Status: ✅ Working
│   │   ├── parser.py                     Purpose: Docling/PyMuPDF parser → ParsedBlock
│   │   │                                 Imported by: ingest_service, build_index
│   │   │                                 Status: ✅ Working
│   │   ├── chunker.py                    Purpose: Section-aware chunking with priority
│   │   │                                 Imported by: ingest_service, build_index, embedder(__main__)
│   │   │                                 Status: ✅ Working
│   │   ├── figure_captioner.py           Purpose: DePlot + Groq figure captioning
│   │   │                                 Imported by: ingest_service, build_index
│   │   │                                 Status: ✅ Working (graceful fallback)
│   │   ├── figure_captioner_groq.py      Purpose: Groq vision captioning fallback
│   │   │                                 Imported by: figure_captioner
│   │   │                                 Status: ✅ Working
│   │   ├── graph_parser.py               Purpose: DePlot chart→table extraction
│   │   │                                 Imported by: figure_captioner
│   │   │                                 Status: ✅ Working (optional)
│   │   ├── table_summarizer.py           Purpose: Ollama table summary
│   │   │                                 Imported by: ingest_service, build_index
│   │   │                                 Status: ⚠️ Requires local Ollama
│   │   └── utils.py                      Purpose: Math preservation, checkpoint, quality filter
│   │                                     Imported by: NOTHING
│   │                                     Status: ⚠️ Dead code — never imported at runtime
│   │
│   ├── retrieval/
│   │   ├── __init__.py                   (empty)
│   │   ├── retriever.py                  Purpose: Hybrid pgvector+FTS retrieval, RRF, Cohere rerank
│   │   │                                 Imported by: rag_service, query(stream), search, pipeline_runner
│   │   │                                 Status: ✅ Working (async path via rag_service)
│   │   ├── embedder.py                   Purpose: SPECTER2 embedding, model caching
│   │   │                                 Imported by: retriever, main(lifespan), build_index, recommender, app.py
│   │   │                                 Status: ✅ Working
│   │   └── hyde.py                       Purpose: Hypothetical Document Embeddings
│   │                                     Imported by: retriever (conditionally)
│   │                                     Status: ⚠️ Requires local Ollama (qwen2.5:7b)
│   │
│   ├── generation/
│   │   ├── __init__.py                   (empty)
│   │   ├── generator.py                  Purpose: Groq/Ollama LLM answer generation
│   │   │                                 Imported by: rag_service, query(stream), pipeline_runner
│   │   │                                 Status: ✅ Working (Groq path)
│   │   └── memory.py                     Purpose: In-memory conversation history
│   │                                     Imported by: rag_service, app.py
│   │                                     Status: ✅ Working
│   │
│   ├── evaluation/
│   │   ├── __init__.py                   (empty)
│   │   ├── evaluator.py                  Purpose: Retrieval metrics + Ollama judge
│   │   │                                 Imported by: run_benchmark (script)
│   │   │                                 Status: ✅ Working (offline script)
│   │   ├── benchmarker.py                Purpose: QA dataset generation
│   │   │                                 Imported by: run_benchmark (script)
│   │   │                                 Status: ✅ Working (offline script)
│   │   ├── retrieval_metrics.py          Purpose: Pure-Python retrieval metric library
│   │   │                                 Imported by: NOTHING at runtime
│   │   │                                 Status: ⚠️ Dead code — complete but never wired in
│   │   ├── generation_metrics.py         Purpose: LLM-as-judge generation metrics
│   │   │                                 Imported by: NOTHING at runtime
│   │   │                                 Status: ⚠️ Dead code — complete but never wired in
│   │   └── pipeline_runner.py            Purpose: Run pipeline over QA dataset
│   │                                     Imported by: run_benchmark (script)
│   │                                     Status: ❌ Broken — calls `retrieve()` with old sync signature
│   │
│   ├── download/
│   │   ├── __init__.py                   (empty)
│   │   ├── arxiv_downloader.py           Purpose: ArXiv paper download CLI
│   │   │                                 Imported by: arxiv_fetcher
│   │   │                                 Status: ✅ Working (standalone CLI)
│   │   └── arxiv_fetcher.py              Purpose: Re-export shim
│   │                                     Imported by: app.py (Streamlit)
│   │                                     Status: ✅ Working
│   │
│   └── ui/
│       ├── __init__.py                   (empty)
│       ├── recommender.py                Purpose: Embedding-based paper recommendations
│       │                                 Imported by: recommend router, app.py
│       │                                 Status: ⚠️ Partially working — requires pre-built index files
│       ├── tts.py                        Purpose: macOS `say` command TTS
│       │                                 Imported by: tts router, app.py
│       │                                 Status: ⚠️ macOS-only — fails silently on Linux/Docker
│       └── flowchart.py                  Purpose: LLM→Mermaid flowchart gen
│                                         Imported by: flowchart router, app.py
│                                         Status: ⚠️ Requires local Ollama
│
├── scripts/
│   ├── build_index.py                    Purpose: Full ingestion + pgvector sync
│   │                                     Status: ✅ Working (offline pipeline)
│   └── run_benchmark.py                  Purpose: Benchmark runner
│                                         Status: ⚠️ Partially working — uses old retriever signature
│
├── app.py                                Purpose: Streamlit UI (2203 lines, 87KB)
│                                         Status: ❌ Broken — imports `detect_query_intent` which doesn't exist
│
├── multimodal/                           Purpose: Planned multimodal pipeline
│                                         Status: ❌ Empty directories — no code
│
├── data/                                 (papers, configs, registry)
├── indexes/                              (pre-built FAISS, BM25, embeddings, metadata)
├── alembic/                              (async migrations)
│
├── Dockerfile                            Status: ❌ Broken — references missing `requirements-cloud.txt` and `start.sh`
├── docker-compose.yml                    Status: ✅ Working (local dev)
├── .env                                  Status: ❌ CRITICAL — contains hardcoded real API keys
├── .env.example                          Status: ⚠️ Incomplete — missing Groq, Cohere, Langfuse keys
└── requirements.txt                      Status: ✅ Present
```

---

## Phase 2: Startup Trace

Starting from [api/main.py](file:///Users/riri/resume_projects/anthology/api/main.py):

### Initialization Order

```
1. Module-level: settings = get_settings()                    [L9]
   → loads .env via pydantic_settings BaseSettings           [config.py:L37-38]
   → lru_cached singleton                                    [config.py:L42-44]

2. Module-level: engine = create_async_engine(...)            [database.py:L21-27]
   → pool_size=1, max_overflow=0 (extremely conservative)    [database.py:L25-26]
   → pool_pre_ping=True                                      [database.py:L24]

3. Module-level: AsyncSessionLocal = async_sessionmaker(...)  [database.py:L29-33]

4. app = FastAPI(lifespan=lifespan)                           [main.py:L28-33]

5. CORS middleware registered                                 [main.py:L36-42]
   → origins: ["http://localhost:3000", "http://localhost:8501"]

6. 10 routers registered                                      [main.py:L45-54]
   → health, query, papers, search, recommend, tts,
     flowchart, benchmark, feedback, stats

7. Root "/" endpoint registered                               [main.py:L57-64]

8. LIFESPAN startup:                                          [main.py:L12-23]
   a. await create_tables()                                   [main.py:L16]
      → Base.metadata.create_all — creates tables if missing
   b. from src.retrieval.embedder import get_model            [main.py:L18]
      get_model() → loads allenai/specter2_base              [embedder.py:L8-13]
   c. print startup message
   d. ALL exceptions during startup are CAUGHT and SWALLOWED  [main.py:L21-22]
      → "Startup warning: {e}" — app starts in degraded state silently
```

> [!CAUTION]
> **Startup swallows ALL exceptions** ([main.py:L21-22](file:///Users/riri/resume_projects/anthology/api/main.py#L21-L22)). If the database is unreachable or the embedding model fails to load, the app starts anyway with no error propagation. This means the health endpoint could report "ok" while the system is fundamentally broken.

### Missing from Startup
- No structured logging setup
- No graceful shutdown logic (just a print)
- No database migration check (relies on `create_tables` which may not match Alembic schema)
- No readiness probe (health endpoint checks pgvector index, not basic connectivity)

---

## Phase 3: Router Audit

### Route Table

| Route | Method | Request Model | Response Model | DB | Service | External |
|---|---|---|---|---|---|---|
| `/` | GET | — | dict | — | — | — |
| `/health` | GET | — | `HealthResponse` | ✅ | — | pgvector |
| `/api/v1/query` | POST | `QueryRequest` | `QueryResponse` | ✅ | `RAGService` | pgvector, Groq/Ollama, Cohere, Redis, Langfuse |
| `/api/v1/query/stream` | POST | `QueryRequest` | StreamingResponse | — | — | pgvector(?), Groq/Ollama |
| `/api/v1/papers` | GET | — | `PaperListResponse` | ✅ | `PaperService` | — |
| `/api/v1/papers/{id}` | GET | UUID | `PaperOut` | ✅ | `PaperService` | — |
| `/api/v1/papers/upload` | POST | UploadFile | dict | ✅ | `ingest_single_paper` | Groq, DePlot |
| `/api/v1/papers/sync` | POST | — | dict | ✅ | `PaperService` | — |
| `/api/v1/vectors/search` | POST | query params | dict | ✅ | — (raw SQL) | PostgreSQL FTS |
| `/api/v1/search` | POST | `SearchRequest` | `SearchResponse` | — | — | FAISS/pgvector |
| `/api/v1/recommend` | POST | `RecommendRequest` | `RecommendResponse` | — | — | ArXiv API, numpy |
| `/api/v1/tts` | POST | `TTSRequest` | audio/wav | — | — | macOS `say` |
| `/api/v1/flowchart` | POST | `FlowchartRequest` | `FlowchartResponse` | — | — | Ollama |
| `/api/v1/benchmark` | GET | — | dict (JSON file) | — | — | filesystem |
| `/api/v1/benchmark/report` | GET | — | dict (JSON file) | — | — | filesystem |
| `/api/v1/feedback` | POST | `FeedbackRequest` | `FeedbackResponse` | ✅ | — | — |
| `/api/v1/feedback/{id}` | GET | query_id | dict | ✅ | — | — |
| `/api/v1/stats` | GET | — | `StatsResponse` | ✅ | `StatsService` | — |

### Broken Routes

#### 1. `/api/v1/query/stream` — **BROKEN**

At [query.py:L27](file:///Users/riri/resume_projects/anthology/api/routers/query.py#L27), the streaming route calls `retrieve()` with the **old synchronous signature**:
```python
chunks = retrieve(request.question, top_k=request.top_k, use_hyde=request.use_hyde)
```
But [retriever.py:L145-153](file:///Users/riri/resume_projects/anthology/src/retrieval/retriever.py#L145-L153) defines `retrieve` as `async def retrieve(query, top_k, db=None, ...)` and **raises `ValueError("db session required")`** if `db` is None. The streaming route never passes `db`. **This will crash at runtime.**

#### 2. `/api/v1/search` — **BROKEN**

Same issue. [search.py:L12-16](file:///Users/riri/resume_projects/anthology/api/routers/search.py#L12-L16) calls `retrieve()` synchronously without `db` and without `await`. The retriever is an `async def`. **Will crash at runtime.**

#### 3. `/api/v1/papers/sync` — **BROKEN**

[paper_service.py:L24](file:///Users/riri/resume_projects/anthology/api/services/paper_service.py#L24) references `settings.chunks_path`, but `chunks_path` is **not defined** in [config.py](file:///Users/riri/resume_projects/anthology/api/core/config.py). Pydantic will raise an `AttributeError` at runtime.

#### 4. `/api/v1/tts` — **NOT PORTABLE**

[tts.py:L96](file:///Users/riri/resume_projects/anthology/src/ui/tts.py#L96) uses `shutil.which("say")`. The `say` command only exists on macOS. On the Render deployment (Linux), this always returns `None` → HTTP 500.

---

## Phase 4: Upload Pipeline Investigation

### `POST /api/v1/papers/upload` — End-to-End Trace

| Step | Implemented | Wired | Reachable | Status |
|---|---|---|---|---|
| 1. Request received | ✅ | ✅ | ✅ | FastAPI UploadFile |
| 2. PDF validation | ✅ | ✅ | ✅ | `.pdf` extension check |
| 3. File storage | ✅ | ✅ | ✅ | Written to `data/papers/` |
| 4. PDF processing | ✅ | ✅ | ❌ | **Never reached due to step 4a** |
| 5. Chunking | ✅ | ✅ | ❌ | Depends on step 4 |
| 6. Embedding | ✅ | ✅ | ❌ | Depends on step 5 |
| 7. Vector indexing | ✅ | ✅ | ❌ | pgvector INSERT |
| 8. Metadata persistence | ⚠️ | — | ❌ | No Paper table row created |
| 9. Background jobs | ❌ | — | — | Everything is synchronous |
| 10. Response | ✅ | ✅ | ❌ | Never reached |

### Critical Bug: Nested Event Loop Deadlock

[papers.py:L33-34](file:///Users/riri/resume_projects/anthology/api/routers/papers.py#L33-L34):
```python
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, lambda: asyncio.run(ingest_single_paper(str(dest), db)))
```

This does `asyncio.run()` **inside** a `run_in_executor()` thread, but passes the **async SQLAlchemy session** (`db`) from the main loop to a new event loop in a different thread. This will:
1. Create a **new event loop** in the executor thread (`asyncio.run()`)
2. Try to use the `db` session which is **bound to the original event loop**
3. **Crash** with `RuntimeError` or `asyncio` errors

> [!CAUTION]
> **Verdict: BROKEN** — The upload pipeline will crash at runtime due to async event loop misuse. The file is written to disk (step 3), but everything after that fails.

### Missing Implementations
- **No Paper table row** is created during upload — only chunks are inserted
- **No background job** — ingestion is fully synchronous and blocks the request
- **No file size validation** — a 500MB PDF will be accepted
- **No duplicate detection** — uploading the same file overwrites silently

---

## Phase 5: RAG Pipeline Audit

### Capability Matrix

| Capability | Claimed (README) | Code Exists | Wired into Runtime | Actually Used | Status |
|---|---|---|---|---|---|
| PDF ingestion | ✅ | ✅ | ✅ (build_index.py) | ✅ (offline) | **Working (offline only)** |
| Section-aware chunking | ✅ | ✅ | ✅ | ✅ | **Working** |
| Metadata extraction | ✅ | ✅ | ✅ | ✅ | **Working** |
| Citation tracking | ✅ | ✅ | ✅ | ✅ | **Working** |
| Embedding generation (SPECTER2) | ✅ | ✅ | ✅ | ✅ | **Working** |
| pgvector dense retrieval | ✅ | ✅ | ✅ | ✅ | **Working** |
| PostgreSQL FTS | ✅ | ✅ | ✅ | ✅ | **Working** |
| Reciprocal Rank Fusion | ✅ | ✅ | ✅ | ✅ | **Working** |
| BM25 (rank_bm25 library) | ✅ (README) | ❌ (no code in API) | — | Only in offline `indexes/bm25_index.pkl` | **Not in runtime** |
| Cross-encoder reranking | ✅ (README) | ✅ [retriever.py:L22-27](file:///Users/riri/resume_projects/anthology/src/retrieval/retriever.py#L22-L27) | ❌ | Never called | **Dead code** |
| Cohere reranking | Not claimed | ✅ [retriever.py:L98-119](file:///Users/riri/resume_projects/anthology/src/retrieval/retriever.py#L98-L119) | ✅ | ✅ | **Working** |
| HyDE query expansion | ✅ | ✅ | ✅ (opt-in) | ✅ | **Working** (requires Ollama) |
| Context building | ✅ | ✅ | ✅ | ✅ | **Working** |
| LLM answer generation | ✅ | ✅ | ✅ | ✅ | **Working** (Groq) |
| Conversation memory | ✅ | ✅ | ✅ | ✅ | **Working** (in-process only) |
| Evaluation framework | ✅ | ✅ | ❌ (offline only) | Script-only | **Working offline** |
| Feedback loops | ✅ | ✅ | ✅ | ✅ (DB storage) | **Working** (no re-training) |
| FAISS retrieval | ✅ (README) | ❌ (removed from API) | — | Only in pre-built index files | **Not in runtime** |
| Multimodal retrieval | ✅ (README) | ❌ | — | — | **Not implemented** |
| Figure analysis | ✅ (README) | ✅ (ingestion) | ✅ (build_index) | Offline only | **Working offline** |
| Streaming responses | ✅ | ✅ [query.py:L21-40](file:///Users/riri/resume_projects/anthology/api/routers/query.py#L21-L40) | ❌ | **Broken** | **Broken wiring** |
| Redis caching | Not in README | ✅ [rag_service.py:L38-45](file:///Users/riri/resume_projects/anthology/api/services/rag_service.py#L38-L45) | ✅ (optional) | ✅ if Redis up | **Working** |
| Langfuse tracing | Not in README | ✅ [rag_service.py:L13-18](file:///Users/riri/resume_projects/anthology/api/services/rag_service.py#L13-L18) | ✅ | ✅ | **Working** |

### Dead Code in Retrieval

The **cross-encoder** model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is loaded via `_get_cross_encoder()` at [retriever.py:L22-27](file:///Users/riri/resume_projects/anthology/src/retrieval/retriever.py#L22-L27) but this function is **never called** anywhere. The runtime uses **Cohere rerank** instead ([retriever.py:L98-119](file:///Users/riri/resume_projects/anthology/src/retrieval/retriever.py#L98-L119)).

---

## Phase 6: Dependency Graph

### Module Dependency Flow

```mermaid
graph TD
    A["api/main.py"] --> B["api/core/config.py"]
    A --> C["api/core/database.py"]
    A --> R1["routers/health"]
    A --> R2["routers/query"]
    A --> R3["routers/papers"]
    A --> R4["routers/search"]
    A --> R5["routers/recommend"]
    A --> R6["routers/tts"]
    A --> R7["routers/flowchart"]
    A --> R8["routers/benchmark"]
    A --> R9["routers/feedback"]
    A --> R10["routers/stats"]
    
    R2 --> S1["services/rag_service"]
    R3 --> S2["services/ingest_service"]
    R3 --> S3["services/paper_service"]
    R10 --> S4["services/stats_service"]
    
    S1 --> RET["src/retrieval/retriever"]
    S1 --> GEN["src/generation/generator"]
    S1 --> MEM["src/generation/memory"]
    
    RET --> EMB["src/retrieval/embedder"]
    RET --> HYD["src/retrieval/hyde"]
    
    S2 --> PAR["src/ingestion/parser"]
    S2 --> CHK["src/ingestion/chunker"]
    S2 --> ING["src/ingestion/ingest"]
    S2 --> EMB
    
    style S5["services/vector_service"] fill:#ff6b6b,stroke:#c0392b
    style DEP["api/dependencies.py"] fill:#ff6b6b,stroke:#c0392b
```

### Issues Found

| Issue | Severity | Location |
|---|---|---|
| **`api/dependencies.py` is dead code** — defines `get_settings_dep()` but no router imports it | Low | [dependencies.py](file:///Users/riri/resume_projects/anthology/api/dependencies.py) |
| **`services/vector_service.py` is dead code** — complete pgvector service, never imported | Medium | [vector_service.py](file:///Users/riri/resume_projects/anthology/api/services/vector_service.py) |
| **`src/ingestion/utils.py` is dead code** — `preserve_math`, `filter_chunks` never called | Low | [utils.py](file:///Users/riri/resume_projects/anthology/src/ingestion/utils.py) |
| **`src/evaluation/retrieval_metrics.py` is dead code** — complete library, never imported at runtime | Low | [retrieval_metrics.py](file:///Users/riri/resume_projects/anthology/src/evaluation/retrieval_metrics.py) |
| **`src/evaluation/generation_metrics.py` is dead code** — complete library, never imported at runtime | Low | [generation_metrics.py](file:///Users/riri/resume_projects/anthology/src/evaluation/generation_metrics.py) |
| **Global mutable state** — `_sessions` dict in [rag_service.py:L24](file:///Users/riri/resume_projects/anthology/api/services/rag_service.py#L24) grows unboundedly | Medium | Memory leak |
| **Global mutable state** — `_registry_cache` in [ingest.py:L19](file:///Users/riri/resume_projects/anthology/src/ingestion/ingest.py#L19) | Low | Singleton |
| **Service locator** — Routers use inline `from ... import` inside route handlers (lazy imports) | Low | search.py, recommend.py, tts.py, flowchart.py |
| **No circular imports detected** | — | — |
| **No dependency injection** — services instantiated at module level as globals | Low | All services |

---

## Phase 7: Database Audit

### ORM Models ([tables.py](file:///Users/riri/resume_projects/anthology/api/models/tables.py))

| Table | Columns | Foreign Keys | Status |
|---|---|---|---|
| `papers` | 13 columns (id, arxiv_id, filename, title, ...) | — | ✅ |
| `queries` | 11 columns (id, question, answer, ...) | — | ✅ |
| `feedback` | 5 columns (id, query_id, rating, comment, ...) | `query_id → queries.id` | ✅ |
| `chunks` | 18 columns (id, chunk_id, source, ..., embedding Vector(768)) | — | ⚠️ No FK to papers |

### Schema Issues

1. **`chunks` has no foreign key to `papers`** — The `source` field stores the filename as a string, matching `papers.filename` by convention, not by constraint. Orphaned chunks are possible.

2. **Dual table creation paths** — [database.py:L48-50](file:///Users/riri/resume_projects/anthology/api/core/database.py#L48-L50) uses `Base.metadata.create_all` at startup, while Alembic migrations also exist. These can conflict — Alembic may expect to own the schema.

3. **`embedding_dim` mismatch** — [stats_service.py:L26](file:///Users/riri/resume_projects/anthology/api/services/stats_service.py#L26) hardcodes `embedding_dim=1024`, but the ORM defines `Vector(768)` at [tables.py:L86](file:///Users/riri/resume_projects/anthology/api/models/tables.py#L86) and SPECTER2 produces 768-dimensional embeddings.

### Alembic Migrations

4 migration files exist in `alembic/versions/`:
- `fcfca052c549_initial_tables.py`
- `5890fefb391a_add_chunks_table_with_pgvector.py`
- `multimodal_columns.py`
- `specter2_migration.py`

> [!WARNING]
> `alembic.ini` line 89 sets `sqlalchemy.url = %(DATABASE_URL)s` which expects a `DATABASE_URL` environment variable, but `alembic/env.py` also reads `DATABASE_URL` from `os.getenv`. The `%(...)s` interpolation in alembic.ini will fail if `DATABASE_URL` is not a ConfigParser variable.

### Connection Pool

[database.py:L25-26](file:///Users/riri/resume_projects/anthology/api/core/database.py#L25-L26):
```python
pool_size=1,
max_overflow=0,
```
This means **exactly 1 database connection** for the entire application. Under any concurrent load, queries will serialize. This is acceptable for a single-user demo but will not scale.

---

## Phase 8: Vector Store Audit

### Active: pgvector (PostgreSQL)

| Aspect | Implementation | Status |
|---|---|---|
| Index creation | `create_tables()` creates the `chunks` table with `Vector(768)` column | ✅ |
| Data insertion | `ingest_service.py` inserts with `::vector` cast | ✅ |
| Query | Cosine distance `<=>` operator in raw SQL | ✅ |
| Persistence | PostgreSQL (Docker volume) | ✅ |
| Index type | **No HNSW or IVFFlat index created** — brute-force scan | ⚠️ |

> [!WARNING]
> No vector index is created. All similarity searches are **brute-force sequential scans** over the embedding column. For ~10K chunks this is acceptable; for larger corpora it will become a bottleneck. Add: `CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)`.

### Legacy: FAISS (offline only)

Pre-built files exist in `indexes/`:
- `faiss_index.bin` (11MB)
- `chunk_embeddings.npy` (39MB)
- `bm25_index.pkl` (4MB)

These are **not loaded by the API** — they're artifacts from the old retrieval pipeline. The API uses pgvector exclusively. They are used by [scripts/run_benchmark.py](file:///Users/riri/resume_projects/anthology/scripts/run_benchmark.py) and the legacy `app.py`.

### Dimension Check
- SPECTER2 (`allenai/specter2_base`) → 768 dimensions ✅
- `Vector(768)` in ORM → matches ✅
- `StatsService` reports `embedding_dim=1024` → **WRONG** ❌

> [!IMPORTANT]
> The Streamlit app ([app.py:L14](file:///Users/riri/resume_projects/anthology/app.py#L14)) loads `BAAI/bge-small-en-v1.5` (384 dimensions) and injects it via `set_model()`, overriding SPECTER2. This means **the Streamlit app uses a DIFFERENT embedding model than the API**, and would produce 384-dim vectors that are incompatible with the 768-dim pgvector column. The Streamlit app is a completely separate, incompatible path.

---

## Phase 9: Configuration Audit

### `.env` — CRITICAL SECURITY ISSUE

[.env](file:///Users/riri/resume_projects/anthology/.env) is committed to git and contains:

| Key | Value | Risk |
|---|---|---|
| `GROQ_API_KEY` | `gsk_lBvHdfyy...` (real key) | 🔴 **CRITICAL** |
| `COHERE_API_KEY` | `2RP7QZ66p6...` (real key) | 🔴 **CRITICAL** |
| `LANGFUSE_SECRET_KEY` | `sk-lf-45d5afdb-...` (real key) | 🔴 **CRITICAL** |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-dc6755bc-...` (real key) | 🟡 Medium |
| `DATABASE_URL` | `postgresql+asyncpg://anthology:anthology@localhost:5432/anthology` | 🟢 Local only |

> [!CAUTION]
> **THREE production API keys are committed in plaintext.** The `.env` file is tracked in git. These keys should be immediately rotated.

### `.env.example` — Incomplete

Missing: `GROQ_API_KEY`, `USE_GROQ`, `GROQ_MODEL`, `COHERE_API_KEY`, `LANGFUSE_*` keys.

### Dockerfile — BROKEN

[Dockerfile:L14](file:///Users/riri/resume_projects/anthology/Dockerfile#L14): `COPY requirements-cloud.txt .` — This file **does not exist** in the repository. The Docker build will fail at this step.

[Dockerfile:L19](file:///Users/riri/resume_projects/anthology/Dockerfile#L19): `RUN chmod +x start.sh` — `start.sh` **does not exist**. Build fails.

### docker-compose.yml

Uses `pgvector/pgvector:pg16` image — correct. Port mapping is `5432:5432`, but `.env` has port `5432` while [config.py](file:///Users/riri/resume_projects/anthology/api/core/config.py#L13) defaults to port `5433`. Mismatch.

---

## Phase 10: Error Handling Audit

### Critical

| Finding | Location | Impact |
|---|---|---|
| **Startup swallows all exceptions** | [main.py:L21-22](file:///Users/riri/resume_projects/anthology/api/main.py#L21-L22) | App starts in unknown state |
| **SQL injection via f-string** | [vector_service.py:L14](file:///Users/riri/resume_projects/anthology/api/services/vector_service.py#L14), [retriever.py:L42-53](file:///Users/riri/resume_projects/anthology/src/retrieval/retriever.py#L42-L53) | `content_type` and `query_vec` interpolated into SQL |
| **Unbounded memory growth** | [rag_service.py:L24](file:///Users/riri/resume_projects/anthology/api/services/rag_service.py#L24) | `_sessions` dict never evicts |

### High

| Finding | Location | Impact |
|---|---|---|
| **Bare `except Exception` with silent `pass`** | [figure_captioner.py:L28](file:///Users/riri/resume_projects/anthology/src/ingestion/figure_captioner.py#L28) | Caption errors silently ignored |
| **No request timeout for LLM calls** | [generator.py:L70-76](file:///Users/riri/resume_projects/anthology/src/generation/generator.py#L70-L76) | Groq client has `max_retries=0` but no timeout |
| **Redis connection never closed on cache miss** | [rag_service.py:L57](file:///Users/riri/resume_projects/anthology/api/services/rag_service.py#L57) | Connection leak if Redis is up but cache misses |
| **Double commit** | [papers.py→get_db](file:///Users/riri/resume_projects/anthology/api/core/database.py#L36-L45) + [rag_service.py:L130](file:///Users/riri/resume_projects/anthology/api/services/rag_service.py#L130) | `rag_service` commits, then `get_db` commits again |

### Medium

| Finding | Location | Impact |
|---|---|---|
| `datetime.utcnow()` deprecated | [tables.py:L26,L42,L54,L87](file:///Users/riri/resume_projects/anthology/api/models/tables.py#L26) | Should use `datetime.now(UTC)` |
| No input size limit on upload | [papers.py:L14-15](file:///Users/riri/resume_projects/anthology/api/routers/papers.py#L14-L15) | Unbounded memory on large PDFs |
| No retry on Cohere rerank | [retriever.py:L118](file:///Users/riri/resume_projects/anthology/src/retrieval/retriever.py#L118) | Falls back silently on any error |
| `hashlib.md5` for cache keys | [rag_service.py:L34](file:///Users/riri/resume_projects/anthology/api/services/rag_service.py#L34) | Not a security issue here, but md5 is deprecated |

### Low

| Finding | Location | Impact |
|---|---|---|
| `import re` duplicated | [hyde.py:L1,L13](file:///Users/riri/resume_projects/anthology/src/retrieval/hyde.py#L1) | Cosmetic |
| No CORS origin for production domain | [config.py:L35](file:///Users/riri/resume_projects/anthology/api/core/config.py#L35) | Only `localhost:3000` and `localhost:8501` |

---

## Phase 11: Production Readiness Scoring

| Category | Score | Reasoning |
|---|---|---|
| **Architecture** | 6/10 | Clean separation between API, retrieval, generation, evaluation. But two incompatible frontends (Streamlit+FastAPI), dead code modules, and the upload pipeline is fundamentally broken. |
| **Code Quality** | 5/10 | Readable, consistent style. But SQL injection vectors, dead code, duplicate imports, hardcoded values, and mixing sync/async patterns. |
| **Reliability** | 3/10 | Three routes crash at runtime. Startup swallows errors. No retries on external calls. Unbounded memory growth in session dict. |
| **Observability** | 4/10 | Langfuse tracing on the query path is good. But no structured logging anywhere — all output is `print()`. No error tracking (Sentry etc). |
| **Security** | 1/10 | Three API keys committed in plaintext. SQL injection via f-string. No auth on any endpoint. No rate limiting. |
| **Scalability** | 2/10 | pool_size=1. No vector index on pgvector. In-process memory (not shared across workers). Synchronous LLM calls block the event loop via run_in_executor. |
| **Maintainability** | 5/10 | Good module structure. But significant dead code (5+ modules). Two incompatible retrieval paths (Streamlit vs API). No tests. |
| **Deployment Readiness** | 2/10 | Dockerfile broken (missing files). No CI/CD. No Kubernetes manifests. CORS only allows localhost. |
| **Testing** | 0/10 | Zero test files. No unit tests, integration tests, or e2e tests. |
| **Documentation** | 4/10 | README is well-written but overclaims. No API documentation beyond Swagger auto-gen. No architecture decision records. |

**Overall: 3.2/10**

---

## Phase 12: Reality Check

| Feature | README Claims | Code Exists | Runtime Reachable | Status |
|---|---|---|---|---|
| Research paper ingestion | ✅ | ✅ | ✅ (offline build_index.py) | **Working (offline)** |
| Citation-grounded QA | ✅ | ✅ | ✅ | **Working** |
| PostgreSQL FTS | ✅ | ✅ | ✅ | **Working** |
| Reciprocal Rank Fusion | ✅ | ✅ | ✅ | **Working** |
| Cross-encoder reranking | ✅ | ✅ (defined) | ❌ (never called) | **Dead code** |
| Cohere reranking | ❌ (not in README) | ✅ | ✅ | **Working (undocumented)** |
| HyDE query expansion | ✅ | ✅ | ✅ (requires Ollama) | **Working (Ollama only)** |
| BM25 | ✅ | ❌ (not in API) | ❌ | **Not in runtime** |
| FAISS dense retrieval | ✅ | ❌ (not in API) | ❌ | **Not in runtime** |
| Hybrid retrieval | ✅ | ✅ (pgvector+FTS) | ✅ | **Working** |
| Local LLM (Ollama) | ✅ | ✅ | ✅ (fallback) | **Working (local only)** |
| API-based LLM (Groq) | ✅ | ✅ | ✅ | **Working** |
| Conversation memory | ✅ | ✅ | ✅ | **Working (in-process)** |
| Evaluation framework | ✅ | ✅ | ❌ (scripts only) | **Working offline** |
| QASPER-style benchmarks | ✅ | ✅ | ❌ | **Working offline** |
| Multimodal document understanding | ✅ ("In Progress") | ❌ | ❌ | **Empty directories** |
| Figure analysis | ✅ | ✅ (ingestion) | ❌ (not in retrieval) | **Ingestion only** |
| Table understanding | ✅ | ✅ (ingestion) | ❌ (not in retrieval) | **Ingestion only** |
| Streaming responses | Implied | ✅ | ❌ | **Broken wiring** |
| Paper upload | Implied | ✅ | ❌ | **Broken (async bug)** |
| Deployed API | ✅ | ✅ | ❌ (Dockerfile broken) | **Cannot build Docker** |
| Streamlit frontend | ✅ | ✅ | ❌ (imports missing function) | **Broken import** |

---

## Phase 13: Final Findings

### 1. Critical Bugs

| # | Bug | Location | Impact |
|---|---|---|---|
| 1 | **API keys committed in plaintext** | [.env](file:///Users/riri/resume_projects/anthology/.env) | Security breach |
| 2 | **Upload pipeline crashes** — nested `asyncio.run()` inside executor with cross-loop db session | [papers.py:L33-34](file:///Users/riri/resume_projects/anthology/api/routers/papers.py#L33-L34) | Upload unusable |
| 3 | **SQL injection** — f-string SQL with unescaped `content_type` | [retriever.py:L42](file:///Users/riri/resume_projects/anthology/src/retrieval/retriever.py#L42), [vector_service.py:L14](file:///Users/riri/resume_projects/anthology/api/services/vector_service.py#L14) | Data breach |
| 4 | **Streamlit app broken** — imports `detect_query_intent` which doesn't exist in `retriever.py` | [app.py:L20](file:///Users/riri/resume_projects/anthology/app.py#L20) | Streamlit won't start |

### 2. Broken Routes

| Route | Issue |
|---|---|
| `POST /api/v1/query/stream` | Calls async `retrieve()` synchronously without `db` |
| `POST /api/v1/search` | Same issue |
| `POST /api/v1/papers/upload` | Nested event loop crash |
| `POST /api/v1/papers/sync` | `settings.chunks_path` undefined |
| `POST /api/v1/tts` | macOS-only, fails on Linux |

### 3. Dead Code

| Module | Lines | Purpose |
|---|---|---|
| `api/dependencies.py` | 9 | Dependency helpers — never imported |
| `api/services/vector_service.py` | 51 | pgvector service — never imported |
| `src/ingestion/utils.py` | 47 | Math preservation, quality filter — never imported |
| `src/evaluation/retrieval_metrics.py` | 382 | Complete retrieval metrics library — never imported at runtime |
| `src/evaluation/generation_metrics.py` | 378 | Complete generation metrics library — never imported at runtime |
| `src/retrieval/retriever._get_cross_encoder()` | ~6 | Cross-encoder model — never called |
| `multimodal/` | 0 | 6 empty directories |

**Total dead code: ~873 lines + 6 empty directories**

### 4. Missing Implementations

- No authentication / authorization
- No rate limiting
- No request validation (file size, content-type verification)
- No health check for external dependencies (Groq, Cohere, Redis)
- No structured logging (all `print()`)
- No test suite
- No CI/CD pipeline
- No pgvector HNSW index
- Missing `requirements-cloud.txt` and `start.sh` for Docker build
- No CORS config for production domain

### 5. Refactoring Priorities

1. **Fix the three broken routes** (search, stream, upload)
2. **Parameterize all raw SQL** — eliminate f-string SQL injection
3. **Add `chunks_path` to Settings** or fix `paper_service.sync_registry_to_db`
4. **Remove dead code** — dependencies.py, vector_service.py, utils.py, cross-encoder
5. **Unify embedding model** — Streamlit uses bge-small, API uses SPECTER2
6. **Add pgvector HNSW index** for scalable similarity search
7. **Increase connection pool** from 1 to at least 5
8. **Add session eviction** to `_sessions` dict to prevent memory leak

### 6. Technical Debt Summary

| Category | Items |
|---|---|
| **Security** | API keys in git, SQL injection, no auth, no rate limiting |
| **Reliability** | Swallowed startup errors, broken routes, no retries |
| **Architecture** | Two incompatible frontends, dead code, dual schema management |
| **Operations** | No tests, no CI, broken Dockerfile, no monitoring |
| **Data integrity** | No FK from chunks→papers, no vector index, dimension mismatch in stats |

### 7. Fastest Path to Production Stability

---

### What works today
- `POST /api/v1/query` — Full RAG pipeline: embed → pgvector + FTS → RRF → Cohere rerank → Groq generation → Langfuse trace → DB persist → Redis cache
- `GET /api/v1/papers` — List papers from DB
- `GET /api/v1/papers/{id}` — Get single paper
- `POST /api/v1/feedback` — Submit feedback
- `GET /api/v1/feedback/{id}` — Get feedback
- `GET /api/v1/stats` — System statistics (wrong embedding_dim)
- `GET /health` — Health check
- `GET /api/v1/benchmark` — Read benchmark JSON
- `POST /api/v1/recommend` — Recommendations (if index files exist)
- Offline pipeline: `scripts/build_index.py` → parse → chunk → embed → sync to pgvector

### What appears to work but actually doesn't
- **Upload pipeline** — File saves, then crashes on async event loop misuse
- **Streaming query** — Route exists but crashes because `retrieve()` is async and needs `db`
- **Search endpoint** — Same async/sync mismatch
- **Paper sync** — Crashes on `settings.chunks_path` AttributeError
- **TTS endpoint** — Returns 500 on any non-macOS system
- **Cross-encoder reranking** — Function defined but never called; Cohere is used instead
- **Streamlit app** — Won't start due to missing `detect_query_intent` import
- **Docker deployment** — Build fails on missing files

### What is completely missing
- Authentication and authorization
- Test suite (unit, integration, e2e)
- CI/CD pipeline
- Structured logging
- Error monitoring (Sentry, etc.)
- pgvector index (HNSW/IVFFlat)
- Multimodal retrieval (empty directories)
- Production Dockerfile
- CORS for production domain
- Request rate limiting
- File upload size limits

### What I would fix first if given one week

**Day 1-2: Stop the bleeding**
1. Rotate all API keys, remove `.env` from git, add to `.gitignore`
2. Fix SQL injection — parameterize all raw SQL queries
3. Fix upload route — remove nested `asyncio.run()`, call `ingest_single_paper` directly as `await`
4. Fix search/stream routes — pass `db` session, properly `await` async retriever
5. Add `chunks_path` to Settings class

**Day 3-4: Stabilize**
6. Fix `stats_service` embedding_dim to 768
7. Add pgvector HNSW index
8. Increase pool_size to 5
9. Add session eviction with max size to `_sessions`
10. Delete dead code (dependencies.py, vector_service.py, etc.)
11. Fix CORS to include production domain

**Day 5: Operations**
12. Create working Dockerfile and start.sh
13. Add basic pytest suite (health, query, feedback endpoints)
14. Add structured logging (replace all `print()` with `logging`)
15. Don't swallow startup exceptions — fail fast
