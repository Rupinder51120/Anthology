# Anthology

**An AI research assistant that lets you search, chat with, and organize a corpus of scientific papers — built on a hybrid dense+sparse retrieval pipeline with a rigorously benchmarked retrieval strategy.**

Anthology ingests PDFs, parses and chunks them with layout/table/figure awareness, embeds them with a scientific-paper-specific model, and serves grounded, cited answers over them through a FastAPI backend and a React frontend. It also includes a real evaluation harness: a 247-question internal benchmark comparing 6 retrieval strategies, and an external retrieval-generalization check against the third-party QASPER dataset.

---

## What problem this solves

Reading and cross-referencing dozens of research papers by hand doesn't scale. Anthology lets you drop a folder of PDFs into a library, then:
- ask natural-language questions and get answers grounded in — and cited from — the actual papers,
- search the corpus semantically instead of by keyword alone,
- organize papers into collections,
- discover new papers on a topic from ArXiv/OpenAlex,
- and quantitatively compare retrieval strategies instead of assuming one "just works."

## Key features

- **Hybrid retrieval**: dense (pgvector cosine similarity) + sparse (Postgres full-text search) fused via Reciprocal Rank Fusion, then reranked with Cohere `rerank-v3.5`
- **Multimodal ingestion**: Docling-based PDF parsing with table/figure enrichment (LLM-generated captions/summaries for non-text content)
- **Section-aware chunking** that preserves short, high-value scientific facts and table/figure context
- **Real-time streaming answers** (SSE) with inline citations, grounded via a lexical-overlap heuristic
- **Swappable generation backend** — Groq (cloud) or Ollama (local) via a config switch, not automatic failover (see [Limitations](#limitations--honest-caveats))
- **Collections, paper-scoped chat, live paper discovery** (ArXiv + OpenAlex), and a self-serve benchmark dashboard
- **A real evaluation harness**, not a single accuracy number: a 6-strategy internal comparison plus an external QASPER retrieval-generalization check

## Architecture

```mermaid
flowchart TD
    PDF["PDF Upload"] --> DOCLING["Docling Parsing"]
    DOCLING --> META["Metadata Extraction"]
    META --> CHUNK["Section-aware Chunking"]
    CHUNK --> ENRICH["Table / Figure Enrichment<br/>(Groq captioning)"]
    ENRICH --> EMBED["SPECTER2 Embeddings"]
    EMBED --> PG[("Postgres + pgvector")]

    subgraph RETRIEVAL["Query-time Retrieval"]
        DENSE["Dense (pgvector cosine)"]
        SPARSE["Sparse (Postgres FTS)"]
        RRF["RRF Fusion"]
        RERANK["Cohere Rerank v3.5"]
        DENSE --> RRF
        SPARSE --> RRF
        RRF --> RERANK
    end

    PG --> RETRIEVAL
    RERANK --> GEN["Groq / Ollama<br/>(swappable, config-selected)"]
    GEN --> STREAM["SSE Streaming + Citations"]
    STREAM --> UI["React Frontend"]

    REDIS[("Redis<br/>response cache")] -.-> RETRIEVAL
    LANGFUSE["Langfuse<br/>(trace/observability)"] -.-> GEN
```

## Retrieval pipeline

1. **Dense retrieval** — the query is embedded with the same SPECTER2 model used for ingestion; pgvector performs cosine-similarity search over `chunks.embedding`.
2. **Sparse retrieval** — Postgres `to_tsvector`/`ts_rank_cd` full-text search, computed at query time (no persisted tsvector column or GIN index yet — acceptable at the current ~12K-chunk scale).
3. **Fusion** — dense and sparse result lists are merged via Reciprocal Rank Fusion (RRF).
4. **Reranking** — the fused candidates are reranked by Cohere `rerank-v3.5` (the production default, `hybrid_rerank`).
5. **Generation** — the top reranked chunks are passed to Groq (cloud, `openai/gpt-oss-20b`) or Ollama (local, `qwen2.5:7b`) depending on the `USE_GROQ` config switch, streamed back over SSE with citations built directly from the chunks actually used in the prompt.
6. **Grounding** — a lexical word-overlap heuristic checks the answer against the retrieved context before it's returned (not semantic/entailment-based — see Limitations).

## Ingestion pipeline

`PDF → Docling parsing → metadata extraction → section-aware chunking → table/figure enrichment (Groq) → SPECTER2 embedding (768-dim) → Postgres/pgvector write`

- Table and figure chunks get LLM-generated captions/summaries so they carry retrievable meaning even though they're not prose.
- Chunking preserves short, information-dense scientific facts that naive fixed-size splitting would fragment.
- Ingestion is transactional per paper — a failure (e.g. a timeout) leaves no partial row in the database.

## Evaluation methodology

Anthology is evaluated two ways, and the two are **not interchangeable**:

**A. Internal benchmark (strategy comparison)** — 247 questions, self-generated from the actual ingested corpus via local Ollama (question generated from a chunk, then scored on whether retrieval finds that chunk). This is a legitimate way to compare retrieval strategies against each other on your own data, but it is self-referential by construction and structurally favors lexically-similar strategies. It answers "which of our strategies retrieves best on our own corpus?" — not "how well does this generalize?"

**B. External benchmark (generalization check)** — retrieval-only evaluation against [QASPER](https://huggingface.co/datasets/allenai/qasper) (Dasigi et al., 2021), a third-party, CC-BY-4.0 dataset of 5,049 questions over 1,585 NLP papers, each written by a practitioner who had only read the paper's title/abstract. This measures how the same embedding/retrieval code performs on a corpus and question style it has never seen. See `scripts/evaluate_qasper.py` for the full methodology and its documented limitations (retrieval-only, no reranking, paragraph-level QASPER-native chunking).

Both are real, reproducible, and checked into the repo (`benchmarks/qa_dataset_v1.json` for A; `scripts/evaluate_qasper.py` for B) — neither is fabricated or hand-picked.

## Verified benchmark results

### A. Internal — 247 questions, full corpus, 6 strategies compared

| Strategy | Paper Hit@1 | Paper Hit@5 | Paper MRR | Chunk Hit@1 | Chunk Hit@5 | Chunk MRR |
|---|---|---|---|---|---|---|
| sparse | 0.301 | 0.632 | 0.436 | 0.142 | 0.444 | 0.261 |
| dense | 0.433 | 0.649 | 0.524 | 0.041 | 0.143 | 0.086 |
| hybrid_rrf | 0.573 | 0.793 | 0.664 | 0.357 | 0.652 | 0.474 |
| dense_rerank | 0.641 | 0.756 | 0.690 | 0.194 | 0.219 | 0.204 |
| **hybrid_rerank (production default)** | **0.773** | **0.858** | **0.817** | **0.575** | **0.725** | **0.643** |

*(`dense_hyde` is excluded from this table — it was only benchmarked on a 10-question sample, too small to report.)*

### B. External — QASPER validation split, 281 papers, 892 answerable questions, retrieval-only

| Strategy | Paper Hit@1 | Paper Hit@5 | Paper MRR |
|---|---|---|---|
| dense (SPECTER2) | 0.128 | 0.230 | 0.178 |
| sparse (BM25) | 0.221 | 0.353 | 0.281 |
| hybrid_rrf | 0.197 | 0.370 | 0.272 |

**Read honestly**: retrieval quality drops substantially on QASPER relative to the internal benchmark, and — notably — plain BM25 lexical search outperforms the SPECTER2 dense embedding here, the reverse of the internal result. This is a real, if slightly deflating, finding: it suggests the internal benchmark's strategy ranking doesn't fully generalize to a harder, externally-authored, abstract-only-derived question style, and that SPECTER2 (tuned for citation-graph similarity) isn't automatically the best embedding choice for this kind of QA-style retrieval. No Cohere reranking or generation quality was evaluated for QASPER (see `scripts/evaluate_qasper.py` for why).

**Not safe to claim**: generation-quality scores (faithfulness/relevance/completeness) for the internal 247-question run — that judge pass was contaminated by a Groq daily-quota exhaustion mid-run and reports a suspicious flat ~0.50 across all three metrics. A handful of older, differently-scaled "BM25 baseline"/"FAISS only"/etc. runs also exist in the live Benchmark UI's history with no corresponding code path in the current pipeline — those are not reproducible and are not cited above.

## Tech stack

**Backend**: FastAPI, SQLAlchemy (async), PostgreSQL 16 + pgvector, Redis, Alembic
**Ingestion/ML**: Docling, SPECTER2 (`sentence-transformers`), Groq, Cohere rerank-v3.5, Ollama
**Frontend**: React, TypeScript, Vite
**Observability**: Langfuse
**Infra**: Docker Compose (API, Postgres, Redis, Ollama, frontend — 5 services)

## Project structure

```
anthology/
├── api/                  # FastAPI app: routers, schemas, services, core config
│   ├── routers/          # papers, search, query, discovery, benchmark, collections, ...
│   ├── services/         # ingest_service, rag_service, paper_service
│   └── core/             # config, database, model registry
├── src/
│   ├── ingestion/        # parser, chunker, metadata_resolver, enrichment
│   ├── retrieval/        # embedder, retriever
│   ├── generation/        # generator (Groq/Ollama, streaming + non-streaming)
│   └── evaluation/       # evaluator, benchmarker, pipeline_runner
├── scripts/
│   ├── run_benchmark.py       # internal 6-strategy benchmark runner
│   └── evaluate_qasper.py     # external QASPER retrieval-generalization eval
├── benchmarks/
│   └── qa_dataset_v1.json     # the 247-question internal QA dataset
├── alembic/              # DB migrations
├── frontend/             # React/TypeScript/Vite app
├── tests/                # pytest suite
└── docker-compose.yml
```

## Setup

```bash
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` and fill in the values you need:

```env
DATABASE_URL=postgresql+asyncpg://anthology:anthology@localhost:5432/anthology
REDIS_URL=redis://localhost:6379

USE_GROQ=false            # true = Groq (cloud); false = Ollama (local) -- a config switch, not automatic failover
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

COHERE_API_KEY=

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://us.cloud.langfuse.com

OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=allenai/specter2_base
COHERE_RERANK_MODEL=rerank-v3.5
```

No key is required to run the app read-only against an already-ingested corpus with `USE_GROQ=false` (Ollama). Groq/Cohere keys are needed for cloud generation and reranking respectively.

## Running with Docker

```bash
docker compose up -d
```

This starts 5 containers: `api` (FastAPI, port 8000), `db` (Postgres+pgvector, port 5432), `redis` (port 6379), `ollama` (port 11434), and `frontend` (React, port 5173). Migrations run via Alembic (see below); the API's Docker image excludes secrets, local datasets, and internal docs from its build context via `.dockerignore`.

Health check: `curl http://localhost:8000/health`

## Running locally (without Docker)

```bash
alembic upgrade head
uvicorn api.main:app --reload
```

API docs: `http://localhost:8000/docs`

## Running tests

```bash
pytest tests/
```

44 tests currently pass. Two test files are excluded from CI (both run and pass locally — see `.github/workflows/tests.yml` for exact reasoning):
- `tests/test_collections_add_paper.py` needs a live, corpus-seeded Postgres instance (run it against the project's own `docker compose` Postgres).
- `tests/test_retrieval_alignment.py` mocks the embedding call but still imports the real `sentence-transformers`/`torch` stack at module load, which would make CI unnecessarily heavy for logic that never touches the model.

## Benchmark instructions

```bash
# Internal 6-strategy benchmark (writes indexes/results_*.json)
python scripts/run_benchmark.py

# External QASPER retrieval-generalization check (bounded pilot by default)
python scripts/evaluate_qasper.py --num-papers 40
python scripts/evaluate_qasper.py --num-papers 281 --split validation   # full validation split
```

Or trigger a live internal-benchmark run from the UI at `/benchmark`.

## Limitations / honest caveats

- **"Swappable provider," not "automatic fallback."** `USE_GROQ` is a static config switch. If Groq fails at request time, the request returns an error — it does not automatically retry against Ollama.
- **Grounding is lexical, not semantic.** The grounding check is a >15%-word-overlap heuristic between the answer and retrieved context, not an entailment/hallucination-detection model.
- **The 247-question internal benchmark is self-generated and self-referential**, not an external/independent benchmark — see Evaluation Methodology above. The QASPER results are the external check.
- **Generation-quality (faithfulness/relevance/completeness) numbers for the internal benchmark are contaminated** (Groq quota exhaustion mid-judging) and are not quoted anywhere in this README.
- **Metadata extraction is heuristic-based**: `arxiv_id`/`doi`/`abstract`/`year` are missing for a meaningful fraction of ingested papers (title/author extraction is ~100%). This doesn't affect retrieval quality, only browsing/citation metadata completeness.
- **No auth, rate limiting, or ANN/GIN indexing** — fine for a local portfolio demo at ~12K chunks; would need addressing before any public-internet deployment or 10x+ corpus growth.
- **Cohere reranking runs on a trial-tier key** (rate-limited); on repeated 429s the code falls back to unranked RRF ordering after 3 retries.

## Future improvements

- A clean, uncontaminated judge-score run (faithfulness/relevance/completeness) once Groq quota allows
- Cohere reranking + a paid-tier evaluation on the QASPER external check (currently dense/sparse/hybrid_rrf only)
- GIN index (FTS) and ivfflat/HNSW index (vectors) once the corpus grows past the low tens of thousands of chunks
- Basic auth + rate limiting for any hosted (non-local) deployment

## License

MIT — see [LICENSE](LICENSE).
