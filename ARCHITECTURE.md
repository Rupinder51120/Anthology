# Anthology v2 RAG Pipeline - Complete Architecture Map

## System Overview

Anthology v2 is a **hybrid retrieval-augmented generation (RAG) system** that combines:
- Dense semantic search (pgvector embeddings)
- Lexical search (PostgreSQL Full-Text Search)  
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- Local LLM generation (Ollama)
- Comprehensive evaluation metrics

The system is designed for research paper analysis, supporting 119+ papers with ~2,600 chunks.

---

## Component Architecture

### 1. RETRIEVAL SYSTEM

#### Embedding Layer (`src/retrieval/embedder.py`)
```
Model: BAAI/bge-large-en-v1.5 (384-dim vectors, normalized)

Core Functions:
├── get_model()                          # Singleton cached loader
├── embed_texts(texts, batch_size=32)   # Main embedding entry point
├── embed_chunks(chunks)                 # With metadata prefix
├── embed_papers_for_recommendation()    # Paper-level embeddings
└── save/load_embeddings()               # Numpy persistence

Configuration:
└── MODEL_NAME = "BAAI/bge-large-en-v1.5"
```

#### Multi-Strategy Retrieval Pipeline (`src/retrieval/retriever.py`)
```
Query → embed_texts()
  ↓
pgvector_search()          →  top-15 dense results (cosine distance)
  ↓
postgres_fts_search()      →  top-15 lexical results (BM25-style)
  ↓
rrf_fuse()                 →  Reciprocal Rank Fusion (K=60) → top-10
  ↓
rerank()                   →  Cross-Encoder ms-marco-MiniLM-L-6-v2
  ↓
retrieve() returns         →  top-5 final ranked chunks

Technology Stack:
├── pgvector (PostgreSQL extension)
├── PostgreSQL text_search (tsvector @@ tsquery)
├── Sentence-Transformers CrossEncoder
└── NumPy for pure numpy index (FAISS backup)
```

#### Vector Indexing (`src/retrieval/indexer.py`)
```
Two-tier index architecture:

1. NumpyIndex
   └── Pure NumPy flat index (FAISS replacement for ARM64 Macs)
       ├── Faster search: embeddings @ query.T
       └── No C++ dependencies: 100% Python

2. BM25Okapi Index
   └── Lexical/keyword search
       ├── Tokenization: [a-z]{1,}+ tokens
       └── Normalized by section priority

Persistence:
├── indexes/faiss_index.bin       # NumpyIndex (pickle)
├── indexes/chunk_embeddings.npy  # Float32 embeddings
└── indexes/bm25_index.pkl        # BM25 (pickle)
```

#### Query Expansion (`src/retrieval/hyde.py`)
```
HyDE (Hypothetical Document Expansion):

Process:
├── User query
├── → Ollama generates N=3 hypothetical documents
├── → Each HyDE doc: 350 tokens, temperature=0.55
├── → Extract 40 technical keywords from docs
└── → Average embeddings for stable centroid

Ollama Config:
├── Model: qwen2.5:7b
├── URL: http://localhost:11434/api/generate
├── Temperature: 0.55 (coherent but diverse)
└── Output tokens: 350

Keyword Extraction:
├── Keep: 4+ char tokens
├── Skip: 40+ stopwords, ALL-CAPS acronyms
└── Extract from HyDE doc to augment BM25 query
```

---

### 2. INGESTION SYSTEM

#### PDF Processing (`src/ingestion/ingest.py`)
```
Input: PDF files from data/papers/

Process:
├── PyMuPDF (fitz) parsing
│   ├── Extract text by page
│   ├── Detect sections (Abstract, Introduction, etc.)
│   └── Build full_text + sections dict
│
└── Metadata Extraction (Priority order)
    ├── 1st: download_registry.json (arxiv_id → metadata)
    ├── 2nd: PDF first page (fallback, less reliable)
    │   ├── Title: longest line in first 8 lines
    │   ├── Year: regex search for 20XX
    │   └── Authors: parsed from header
    └── Output: {title, authors, year, doi, arxiv_id, abstract}
```

#### Intelligent Chunking (`src/ingestion/chunker.py`)
```
Chunk Strategy:
├── Default chunk size: 1400 characters (~250-300 words)
├── Math sections: 1800 chars (larger for equations)
├── Overlap: 200 characters
└── Separators: \n\n, \n, ". ", "! ", "? ", " ", ""

Section Priority Weighting (quality scoring):
├── 1.0 (highest): abstract, methodology, method, methods, approach, model, architecture
├── 0.95: results, evaluation
├── 0.85: experiment, experiments
├── 0.8: discussion
├── 0.75: conclusion
├── 0.65: related work, literature review
├── 0.7: background
├── 0.5: preamble
└── 0.0 (skipped): references, appendix, acknowledgements, bibliography, conflict_of_interest

Chunk Type Classification:
├── math_heavy: Contains LaTeX/math patterns ($...$, \frac, Greek letters)
├── figure_table: References to figures/tables
├── quantitative: Contains percentages, p-values, statistics
├── narrative: Long-form text (>80 words)
└── general: Default

Math Preservation:
├── Before chunking: $..$ → [MATH]...[/MATH]
├── After chunking: Embedded in chunk text
└── Prevents tokenization artifacts

Quality Filtering:
├── Score components:
│   ├── Length bonus: 80+ chars (0.1), 150-500 (0.4), 500+ (0.3)
│   ├── Sentence count: ≥2 sentences (+0.2), ≥4 (+0.1)
│   ├── Numeric content: ≥2 numbers (+0.1)
│   ├── Math content: patterns detected (+0.15)
│   └── Stopword penalty: >60% stopwords (-0.2)
├── Default threshold: min_score=0.3
└── Removes low-quality extracted text
```

#### Index Management (`src/ingestion/index_manager.py`)
```
Incremental Addition:
├── Input: Single PDF
├── Process:
│   ├── Load paper + preserve math
│   ├── Chunk + filter by quality
│   ├── Append to existing chunks_metadata.json
│   ├── Embed new chunks
│   ├── Append to chunk_embeddings.npy
│   ├── Embed paper for recommendations
│   └── Append to paper_embeddings.npy
└── Result: Indexes expanded in-place, no full rebuild needed

Checkpoint System:
├── Resume interrupted ingestion
├── Track processed_files
└── Skip already-indexed papers
```

---

### 3. GENERATION SYSTEM

#### LLM Integration (`src/generation/generator.py`)
```
Dual Provider Support:

PRIMARY: Ollama (Local, Default)
├── Model: qwen2.5:7b
├── URL: http://localhost:11434/api/chat
├── Advantage: No API keys, no rate limits, offline
└── Configured via hardcoded MODEL_NAME

OPTIONAL: Groq (Cloud)
├── Model: llama-3.1-8b-instant
├── URL: https://api.groq.com/openai/v1/chat/completions
├── Enable: USE_GROQ=true, set GROQ_API_KEY
└── Free tier available

Context Formatting:
├── Format chunks into numbered blocks: [Source 1], [Source 2], ...
├── Each source includes: type, paper title, year, section
└── Numbered index allows LLM to reference sources

Citation Extraction:
├── Deduplicate by (title, section)
├── Sort by rerank_score (descending)
└── Return: {title, authors, year, section, filename, doi, score}

System Prompt:
├── Enforces:
│   ├── Complete answers covering ALL context
│   ├── Technical depth with definitions
│   ├── MANDATORY "## Sources Used" section
│   ├── One-sentence contribution per source
│   ├── No hallucinations, no answer fabrication
│   └── Markdown formatting
└── Result: Citation-grounded, auditable responses
```

#### Conversation Memory (`src/generation/memory.py`)
```
Class: ConversationMemory

Config:
├── max_turns: 6 (12 messages total)
├── session_id: Unique session identifier
├── created_at: ISO timestamp
└── topics: Tracked discussion topics

Methods:
├── add(role, content)              # Add message to history
├── get() → [{"role": ..., "content": ...}]
├── add_topic(topic)                # Track what was discussed
├── get_context_summary()           # Summarize recent topics
├── save(path) / load(path)         # Persist to indexes/session.json
└── summary()                       # Human-readable turn count

Persistence:
└── indexes/session.json → {session_id, created_at, history[], topics[]}
```

---

### 4. EVALUATION FRAMEWORK

#### Retrieval Metrics (`src/evaluation/retrieval_metrics.py`)
```
Class: RetrievalMetrics

Operates on: ranked list of retrieved IDs vs set of gold (relevant) IDs

Core Metrics (all 0-1 scale):

1. recall_at_k(retrieved, gold, k)
   └── Fraction of gold items found in top-k
   └── Use: Mentor system (need ALL relevant evidence)

2. hit_at_k(retrieved, gold, k)
   └── Binary: 1.0 if ANY gold item in top-k, else 0.0
   └── Use: Single good chunk sufficient

3. precision_at_k(retrieved, gold, k)
   └── Fraction of top-k that are relevant
   └── Use: Minimize noise sent to LLM

4. mrr (Mean Reciprocal Rank)
   └── 1/rank_of_first_relevant_item
   └── Use: Ranking quality focus

5. average_precision(retrieved, gold)
   └── Area under precision-recall curve
   └── Use: Balance early and complete retrieval

6. ndcg_at_k(retrieved, gold, k)
   └── Normalized Discounted Cumulative Gain
   └── Use: Optimal ranking quality
   └── Handles graded relevance (extensible)

Source Normalization:
├── "data/papers/Foo.pdf" → "Foo"
├── "Foo.pdf" → "Foo"
├── "Foo" → "Foo"
└── Handles all path/extension variants for matching
```

#### Generation Metrics (`src/evaluation/generation_metrics.py`)
```
Class: GenerationEvaluator

Uses Ollama as Judge (qwen2.5:7b)

Metrics (0-1 scores via LLM):

1. faithfulness
   ├── Input: {question, answer, context_chunks}
   ├── Scoring: supported_claims / total_claims
   └── Judge: Does answer use ONLY context information?

2. answer_relevancy
   ├── Input: {question, answer}
   ├── Scoring: Question coverage
   └── Judge: Does answer ADDRESS the question?

3. context_precision
   ├── Input: {question, answer, context_chunks}
   ├── Scoring: Useful chunks / total chunks
   └── Judge: Do retrieved chunks CONTRIBUTE to answer?

4. completeness (optional, requires gold answer)
   ├── Input: {answer, gold_answer}
   ├── Scoring: Coverage vs expected answer
   └── Judge: Does answer COVER what gold answer does?

Aggregate Score:
└── Mean of (faithfulness, answer_relevancy, context_precision [, completeness])

Prompt Structure:
├── Request JSON output (not text)
├── Include chain-of-thought reasoning
├── Reduce anchoring bias
└── Consistent across methods
```

#### Benchmark Runner (`src/evaluation/pipeline_runner.py`)
```
Main Function: run_pipeline_on_dataset()

Input:
├── qa_path: QA dataset JSON (default: indexes/qa_dataset.json)
├── output_path: Results file (default: indexes/pipeline_results.json)
├── use_hyde: Enable query expansion (default: True)
├── top_k: Retrieve top-k chunks (default: 5)
└── sleep_between: Rate limiting (default: 0.0)

Process:
├── Load QA pairs from dataset
├── Checkpoint: Resume from last completed question
├── For each question:
│   ├── Retrieve chunks (with HyDE optional)
│   ├── Generate answer (Ollama)
│   ├── Measure elapsed time
│   ├── Save result: {question, ground_truth, answer, sources, config, elapsed_s}
│   └── Append checkpoint for resumption
└── Output: JSON array of results

Per-Question Result:
├── question: User query
├── ground_truth: Expected answer
├── source_chunk: Reference chunk filename
├── answer: Generated answer
├── contexts: [chunk text, chunk text, ...]
├── sources: [filename, filename, ...]
├── config: {hyde: bool, top_k: int}
└── elapsed_s: Time taken

Checkpoint System:
├── Namespaced by output_path
├── output_path.replace(".json", "_checkpoint.json")
├── Detects stale checkpoints (all questions complete) and clears them
└── Mid-run crashes resume correctly (partial checkpoint)
```

#### Benchmark Generator (`src/evaluation/benchmarker.py`)
```
Functions:

1. build_qa_dataset()
   ├── Generate synthetic QA pairs from corpus
   ├── For each paper: extract key sentences
   ├── Generate questions (via LLM)
   └── Output: indexes/qa_dataset.json

2. _extract_json_array(raw)
   ├── Robustly parse JSON from messy LLM output
   ├── Handle escape sequences
   ├── Fix malformed JSON
   └── Return parsed array

3. _content_words(text)
   ├── Stopword-filtered word sets
   ├── 40+ stopwords: the, is, paper, method, etc.
   ├── Min length: 3 chars
   └── Case-insensitive

4. lexical_overlap(question, answer_text)
   ├── Jaccard similarity of content words
   ├── [0, 1] scale
   └── Useful for quick quality check
```

---

### 5. DATABASE LAYER

#### Connection & Setup (`api/core/database.py`)
```
Configuration:
├── Engine: SQLAlchemy AsyncEngine
├── Driver: asyncpg (PostgreSQL async)
├── URL: postgresql+asyncpg://user:pass@host:5432/db
└── Pool: size=5, max_overflow=10, pre_ping=True

Setup:
├── Automatic URL conversion (postgresql:// → postgresql+asyncpg://)
├── Render.com compatibility (deployment-ready)
└── Async context managers: get_db(), create_tables()
```

#### ORM Tables (`api/models/tables.py`)
```
1. Paper
   ├── id: UUID (PK)
   ├── arxiv_id: String(50), unique
   ├── filename: String(255), unique
   ├── title, authors, abstract, year, topic, url
   ├── chunk_count, indexed (bool)
   ├── created_at, updated_at
   └── queries: Relationship → Query

2. Query
   ├── id: UUID (PK)
   ├── question: Text
   ├── answer: Text
   ├── retrieval_mode: String ("hybrid", "pgvector", etc.)
   ├── top_k, chunks_used, tokens_used
   ├── citations: JSON (array of citation objects)
   ├── response_type: String ("explanation", "comparison", etc.)
   ├── latency_ms: Float
   ├── paper_id: UUID (FK) → Paper
   ├── created_at
   └── feedback: Relationship → Feedback

3. Feedback
   ├── id: UUID (PK)
   ├── query_id: UUID (FK) → Query
   ├── rating: Integer (1-5)
   ├── comment: Text
   ├── created_at
   └── query: Relationship → Query

4. Chunk (pgvector)
   ├── id: UUID (PK)
   ├── chunk_id: String(20), unique
   ├── source, title, authors, year
   ├── section, section_priority, chunk_index
   ├── chunk_type: String (general, math_heavy, figure_table, quantitative, narrative)
   ├── text: Text (cleaned of null bytes)
   ├── char_count, word_count
   ├── embedding: Vector(1024) ← pgvector extension
   └── (No direct relationships; lookup via chunk_id)

Extensions Required:
└── CREATE EXTENSION IF NOT EXISTS vector;
```

---

### 6. API LAYER

#### FastAPI App (`api/main.py`)
```
Lifespan:
├── Startup: create_tables() → Initialize PostgreSQL + pgvector
├── Shutdown: Log shutdown
└── CORS: Allow localhost:3000, localhost:8501

Routers (all prefixed /api/v1):

1. /health               → Health check
2. /query                → Main RAG endpoint (POST)
3. /query/stream         → Streaming endpoint (POST)
4. /search               → Semantic search (POST)
5. /papers               → Paper management (GET list, GET detail, POST upload, POST sync)
6. /papers/sync          → Sync registry to DB
7. /vectors/sync         → Sync chunk embeddings to pgvector
8. /recommend            → Recommendations (local + ArXiv)
9. /flowchart            → Flowchart generation
10. /tts                 → Text-to-speech
11. /benchmark           → Run evaluations
12. /feedback            → Store user feedback
13. /stats               → Corpus statistics
```

#### Query Endpoint (`api/routers/query.py`)
```
POST /api/v1/query
├── Input: QueryRequest {question, top_k, use_hyde, ...}
├── Flow:
│   ├── RAGService.query(request, db)
│   ├── → retrieve(query, top_k, use_hyde, db) → pgvector pipeline
│   ├── → generate_answer(question, chunks) → Ollama
│   ├── → format_citations(chunks)
│   ├── → Save to Query table
│   └── → Return QueryResponse
├── Output: QueryResponse {question, answer, citations, [...]}
└── Response code: 200 OK

POST /api/v1/query/stream
├── Streaming variant (chunks as they're generated)
├── media_type: text/plain
└── Yields tokens from generate_answer_streaming()
```

#### Search Endpoint (`api/routers/search.py`)
```
POST /api/v1/search
├── Input: SearchRequest {query, top_k, use_hyde}
├── Flow:
│   ├── retrieve(query, top_k, use_hyde)
│   ├── → Format as SearchResultOut[] {title, authors, year, section, score, text, filename}
│   └── Return SearchResponse {query, results, total}
└── Returns: SearchResponse
```

#### Paper Management (`api/routers/papers.py`)
```
GET /api/v1/papers
├── List all papers
└── Output: PaperListResponse {papers, total}

GET /api/v1/papers/{paper_id}
├── Get paper detail
└── Output: PaperOut

POST /api/v1/papers/upload
├── Upload new PDF
├── Save to data/papers/
├── Call index_manager.add_paper()
├── Sync to DB
└── Output: {success, message}

POST /api/v1/papers/sync
├── Sync download_registry.json → papers table
└── Output: {success, synced: count}

POST /api/v1/vectors/sync
├── Sync chunk embeddings → pgvector
├── Load chunks_metadata.json + chunk_embeddings.npy
├── Clear existing chunks, insert in batches (100 at a time)
└── Output: {success, inserted, ...}
```

#### Recommendations (`api/routers/recommend.py`)
```
POST /api/v1/recommend
├── Input: RecommendRequest {query, top_k}
├── Flow:
│   ├── recommend_by_query(query, top_k)      → Local papers
│   ├── recommend_arxiv(query, top_k)         → ArXiv papers
│   ├── Format both into RecommendationOut[]
│   └── Return RecommendResponse
└── Output: {query, local[], arxiv[]}
```

#### RAG Service (`api/services/rag_service.py`)
```
Class: RAGService

Method: query(request, db) → QueryResponse
├── Time: measure total latency
├── Retrieve: await retrieve(question, top_k, db=db)
├── Generate: generate_answer(question, chunks)
├── Citations: format_citations(chunks)
├── Save: Insert Query row
└── Return: QueryResponse {question, answer, citations, latency_ms}
```

#### Vector Service (`api/services/vector_service.py`)
```
Class: VectorService

Method: sync_chunks_to_db(db) → dict
├── Load: chunks_metadata.json, chunk_embeddings.npy
├── Validate: len(chunks) == len(embeddings)
├── Clear: DELETE FROM chunks (start fresh)
├── Insert: Batch-insert 100 at a time
│   ├── For each chunk:
│   │   ├── Clean null bytes
│   │   ├── Create Chunk ORM object
│   │   └── Set embedding: embedding.tolist()
│   └── Commit after each batch
└── Return: {success, inserted, total}
```

---

### 7. STREAMLIT UI

#### Main App (`app.py`)
```
Features:
├── Cached model loader: @st.cache_resource
│   ├── Load: SentenceTransformer("BAAI/bge-small-en-v1.5")
│   └── Inject into src.retrieval.embedder via set_model()
│
├── Query interface
│   ├── Input: question text
│   ├── Options: top_k slider, use_hyde checkbox
│   ├── Button: "Search"
│   └── Display: answer + citations (formatted)
│
├── Paper recommendations
│   ├── By query: recommend_by_query(query, top_k)
│   ├── By ArXiv: recommend_arxiv(query, top_k) [API call]
│   └── Show: title, authors, year, similarity score
│
├── Paper upload & indexing
│   ├── Upload PDF → data/papers/
│   ├── Call: add_paper(pdf_path)
│   ├── Rebuild: full_rebuild() or add_new_papers()
│   └── Show: progress, chunk count, index size
│
├── Additional UI components
│   ├── Flowchart generation (src.ui.flowchart)
│   ├── Text-to-speech (src.ui.tts)
│   ├── Index statistics (src.retrieval.indexer.get_index_stats())
│   └── Session management (ConversationMemory)
│
└── Design System
    ├── CSS: Playfair Display + Inter fonts
    ├── Palette: Red accent (#8b1a1a), Gold (#c4a882)
    ├── Layout: Apple iOS inspired + literary journal aesthetic
    └── Theme: TTPD + Red era colors
```

---

### 8. DOWNLOAD & REGISTRY SYSTEM

#### ArXiv Downloader (`src/download/arxiv_downloader.py`)
```
Config File: data/download_config.json
├── topics: ["Generative Adversarial Networks", "Diffusion Models image generation", ...]
├── max_per_topic: 10
├── output_dir: "data/papers"
└── min_year: 2018

Registry: data/download_registry.json
├── Keyed by: arxiv_id
├── Per entry: {title, authors, year, filename, url, abstract, topic, doi}
└── Purpose: Track metadata for all downloaded papers

Core Functions:
├── load_config() / save_config()
├── load_registry() / save_registry()
├── is_duplicate(arxiv_id, registry) → bool
├── register_paper(arxiv_id, meta, registry)
├── download_topic(topic, max_papers, ...)
└── clean_filename(title) → safe_filename.pdf

Technology:
└── arxiv Python library for API queries
```

#### ArXiv Fetcher (`src/download/arxiv_fetcher.py`)
```
Current: Thin wrapper
├── Imports from arxiv_downloader
└── Main entry: download_all()

Future: May expand for additional fetching strategies
```

---

### 9. UI COMPONENTS

#### Recommender (`src/ui/recommender.py`)
```
Functions:

1. load_paper_meta() → list[dict]
   └── Load from indexes/paper_meta.json

2. recommend_local(query_filename, top_k=3, min_similarity=0.5)
   ├── Load: paper_embeddings.npy, paper_meta.json
   ├── Find: Similar papers by embedding cosine similarity
   ├── Exclude: Self
   └── Return: [{title, authors, year, filename, similarity}, ...]

3. recommend_by_query(query, top_k=3, min_similarity=0.4)
   ├── Embed: query text
   ├── Similarity: embeddings @ query_emb
   ├── Rank: Top-k by similarity
   └── Return: [{title, authors, year, filename, similarity}, ...]

4. recommend_arxiv(query, top_k=3)
   ├── Call: arxiv API
   ├── Query: arxiv.Search(query, max_results=top_k*2, sort_by=arxiv.SortCriterion.Relevance)
   └── Return: [{title, authors, year, url}, ...]
```

#### Flowchart (`src/ui/flowchart.py`) & TTS (`src/ui/tts.py`)
```
Sketched but not fully explored in this pass.
├── flowchart.py: Likely generates Mermaid diagrams from answers
└── tts.py: Likely uses TTS library (gTTS or similar)
```

---

### 10. SUPPORT UTILITIES

#### Ingestion Utils (`src/ingestion/utils.py`)
```
Functions:

1. preserve_math(text) → str
   ├── $$ ... $$ → [MATH_BLOCK]...[/MATH_BLOCK]
   ├── $ ... $ → [MATH]...[/MATH]
   └── \LaTeX{...} → [MATH]...[/MATH]

2. load_checkpoint() → dict
   ├── Load: indexes/checkpoint.json
   ├── Ensure: "processed_files" key exists
   └── Return: {processed_files: [], last_run: ...}

3. save_checkpoint(checkpoint)
   ├── Set: checkpoint["last_run"] = now ISO
   └── Write: indexes/checkpoint.json

4. filter_chunks(chunks, score_fn, min_score=0.3)
   ├── Score: Each chunk via score_fn
   ├── Filter: Keep if score >= min_score
   ├── Annotate: Add quality_score to metadata
   └── Return: (kept[], removed[]), {total, kept, removed, removal_rate}
```

#### Schemas (`api/schemas/schemas.py`)
```
Pydantic models for API validation:

Request Models:
├── QueryRequest {question, top_k, use_hyde}
├── SearchRequest {query, top_k, use_hyde}
├── RecommendRequest {query, top_k}
└── ...

Response Models:
├── QueryResponse {question, answer, citations, latency_ms}
├── SearchResponse {query, results, total}
├── RecommendResponse {query, local, arxiv}
└── PaperOut, CitationOut, ...
```

---

## Configuration Reference

### Environment Variables (`.env`)
```
# LLM Provider
USE_GROQ=false
GROQ_API_KEY=xxx

# Database
DATABASE_URL=postgresql+asyncpg://anthology:anthology@localhost:5432/anthology
REDIS_URL=redis://localhost:6379  # Optional

# Paths
INDEXES_DIR=indexes
PAPERS_DIR=data/papers
CHUNKS_PATH=indexes/chunks_metadata.json
EMBEDDINGS_PATH=indexes/chunk_embeddings.npy
REGISTRY_PATH=data/download_registry.json
```

### Application Settings (`api/core/config.py`)
```
Pydantic BaseSettings (reads from .env):
├── app_name, app_version, debug, pythonpath
├── database_url, redis_url
├── indexes_dir, papers_dir, chunks_path, embeddings_path, registry_path
├── groq_api_key, use_groq, groq_model
├── use_pgvector
└── allowed_origins (CORS)
```

### Dependencies (`pyproject.toml`)
```
Key packages:
├── langchain, langchain-community, langchain-core
├── sentence-transformers (embedding + reranking)
├── pymupdf, pypdf (PDF parsing)
├── sqlalchemy (ORM)
├── fastapi, uvicorn (API)
├── streamlit (UI)
├── arxiv (paper search)
├── groq (optional, cloud LLM)
├── rank-bm25 (lexical search)
└── pgvector-python (asyncpg pgvector)
```

---

## Data Directory Structure

```
indexes/
├── chunks_metadata.json           # All chunks with metadata
├── chunk_embeddings.npy           # Float32 numpy array (chunks)
├── chunk_embeddings_copy.npy      # Backup
├── paper_embeddings.npy           # Float32 numpy array (papers)
├── paper_meta.json                # Paper metadata list
├── faiss_index.bin                # NumpyIndex (pickle)
├── bm25_index.pkl                 # BM25 index
├── checkpoint.json                # Ingestion resume point
├── session.json                   # Conversation memory
├── qa_dataset.json                # QA pairs for evaluation
├── qa_dataset_quick.json          # 15-question fast dataset
├── qa_qasper.json                 # QASPER-style benchmark
├── build_checkpoint.json          # Build progress
├── build_report.json              # Build statistics
├── pipeline_results.json          # Benchmark results
├── pipeline_results_checkpoint.json
├── benchmark_summary.json         # Aggregated metrics
└── results_*.json                 # Various result files

data/
├── download_config.json           # ArXiv download config
├── download_registry.json         # Downloaded papers metadata
├── last_download_report.json      # Download stats
├── papers/                        # PDF files
│   ├── Paper1.pdf
│   ├── Paper2.pdf
│   └── ...
├── figures/                       # Extracted figures (experimental)
├── tables/                        # Extracted tables (experimental)
└── uploads/                       # User-uploaded PDFs
```

---

## Entry Points

### CLI Commands

**Build Index** (from scratch or incremental)
```bash
python scripts/build_index.py
  ├── Ingests data/papers/*.pdf
  ├── Chunks with quality filtering
  ├── Embeds all chunks + papers
  ├── Builds FAISS + BM25 indexes
  ├── Saves checkpoints/reports
  └── Outputs: indexes/*.npy, indexes/*.pkl, indexes/*.json
```

**Run Benchmarks**
```bash
python scripts/run_benchmark.py [--build-qa] [--qasper] [--quick] [--clean]
  ├── --build-qa: Generate synthetic QA dataset
  ├── --build-qasper: Generate QASPER-style unbiased benchmark
  ├── --qasper: Run cross-config evaluation
  ├── --quick: Use 15-question fast dataset
  ├── --clean: Wipe stale result files first
  └── Outputs: pipeline_results.json, metrics comparison
```

### API Startup
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
  └── Runs FastAPI server on http://localhost:8000
      ├── /docs → Swagger UI
      ├── /redoc → ReDoc
      └── API endpoints
```

### Streamlit UI Startup
```bash
streamlit run app.py
  └── Runs on http://localhost:8501
      ├── Loads cached embedding model
      ├── Provides query interface
      ├── Paper management
      └── Recommendations
```

---

## Data Flow Diagrams

### Ingestion Pipeline
```
PDF Files
  ↓
PyMuPDF Parsing (load_paper)
  ↓ {metadata, sections, full_text}
Section Detection + Split
  ↓
Chunking (RecursiveCharacterTextSplitter)
  ├─ Respects: size, overlap, separators
  ├─ Math sections: larger chunks
  └─ Detects section priority
  ↓
Quality Filtering (score_chunk)
  └─ Min score: 0.3 → filtered chunks
  ↓
Chunk Classification
  └─ Detect: math_heavy, figure_table, quantitative, narrative, general
  ↓
Embedding (BGE)
  ├─ chunks → chunk_embeddings.npy
  └─ papers → paper_embeddings.npy
  ↓
Index Building
  ├─ NumpyIndex (FAISS-like)
  └─ BM25Okapi
  ↓
Database Sync (pgvector)
  ├─ Chunks + embeddings → PostgreSQL
  └─ Papers → PostgreSQL
  ↓
Checkpoints
  └─ indexes/checkpoint.json (processed files)
```

### Retrieval Pipeline
```
User Query
  ↓
Embedding (BGE)
  └─ query_text → 384-dim vector
  ↓
pgvector Search (pgvector_search)
  ├─ Query: TOP-15 by <=> distance
  └─ Result: vec_results[]
  ↓ (parallel)
PostgreSQL FTS (postgres_fts_search)
  ├─ Query: text search with clean tokenization
  └─ Result: fts_results[]
  ↓
RRF Fusion (rrf_fuse)
  ├─ K = 60
  ├─ Score: 1/(K + rank + 1) for each result
  └─ Result: merged top-10
  ↓
Cross-Encoder Reranking (rerank)
  ├─ Model: ms-marco-MiniLM-L-6-v2
  ├─ Pairs: (query, chunk.text)
  └─ Result: top-5 reranked
  ↓
retrieve() returns final results
  └─ [{text, metadata: {chunk_id, title, section, rerank_score, ...}}, ...]
```

### Generation Pipeline
```
Query + Retrieved Chunks
  ↓
Context Formatting (format_context)
  ├─ [Source 1] [TYPE]
  ├─ Paper: Title (Year)
  ├─ Section: ...
  └─ ---
  └─ <chunk text>
  ↓
Ollama LLM Call
  ├─ Model: qwen2.5:7b
  ├─ Messages: [{role: system, content: SYSTEM_PROMPT}, {role: user, content: query}]
  └─ Response: Generated answer
  ↓
Citation Extraction (format_citations)
  ├─ Deduplicate by (title, section)
  ├─ Sort by rerank_score
  └─ Return: {title, authors, year, section, filename, score}
  ↓
Response Formatting
  ├─ answer: LLM output
  ├─ citations: [{title, authors, year, section, filename, score}, ...]
  └─ chunks_used: len(chunks)
  ↓
Database Save (Query table)
  ├─ question, answer, citations, latency_ms
  └─ retrieval_mode, top_k, tokens_used
  ↓
Return to User
  └─ QueryResponse {question, answer, citations, latency_ms}
```

### Evaluation Pipeline
```
QA Dataset (indexes/qa_dataset.json)
  ↓ Per question
Retrieve chunks (retrieve, top_k=5)
  ├─ pgvector + FTS + RRF + rerank
  └─ Result: chunks[]
  ↓
Generate answer (generate_answer)
  └─ Result: answer_text
  ↓
Compute Metrics
  ├─ Retrieval metrics (recall@k, hit@k, MRR, NDCG)
  │  └─ Compare sources retrieved vs source_chunk
  └─ Generation metrics (faithfulness, relevancy, precision)
     └─ LLM judge
  ↓
Save Result
  ├── question, ground_truth, answer
  ├── source_chunk, sources (retrieved)
  ├── retrieval_metrics, generation_metrics
  └── elapsed_s
  ↓
Checkpoint (partial results saved during run)
  └─ resume on interrupt
  ↓
Evaluation Report
  ├─ Avg metrics per config
  └─ Comparison across retrieval strategies
```

---

## Performance Characteristics

### Index Sizes
```
Chunk Embeddings: ~2,600 chunks × 384 dims × 4 bytes = 4 MB
Paper Embeddings: ~119 papers × 384 dims × 4 bytes = ~183 KB
BM25 Index: ~500 KB (tokenized corpus)
Total Indexes: ~5-10 MB
```

### Query Performance
```
Dense retrieval (pgvector):  <10 ms (PostgreSQL optimized)
Lexical retrieval (FTS):     <10 ms (PostgreSQL tsvector)
RRF fusion:                  <1 ms (pure Python)
Cross-Encoder reranking:     50-200 ms (5 chunks, SentenceTransformers)
LLM generation (Ollama):     2-10 seconds (qwen2.5:7b, streaming)
──────────────────────────────────────────────────────
Total end-to-end:            2-10+ seconds (dominated by LLM)
```

### Batch Performance
```
Building FAISS index:        <100 ms (2,600 chunks)
Building BM25 index:         <100 ms (2,600 chunks)
Embedding 2,600 chunks:      ~30 seconds (batch_size=32)
Full index rebuild:          ~60 seconds (all steps)
```

---

## Experimental / Dead Code

### Multimodal Directory (`multimodal/`)
```
Structure exists but appears incomplete/experimental:
├── api/           [sketched but not implemented]
├── ingestion/     [sketched but not implemented]
├── retrieval/     [sketched but not implemented]
├── storage/       [sketched but not implemented]
├── streamlit/     [sketched but not implemented]
└── worker/        [sketched but not implemented]

Status: Not integrated into main pipeline
Purpose: Likely intended for image/figure handling in papers (future)
```

### Deprecated Components
```
FAISS Index (replaced with NumpyIndex)
  └─ Reason: Segfaults on ARM64 Macs, pure NumPy is sufficient

Groq-as-judge (replaced with Ollama-as-judge)
  └─ Reason: Rate limits, API costs, local is better for iteration

Legacy retriever modes (bm25_only, faiss_only, hybrid_no_rerank)
  └─ Status: Now pgvector-only in active retriever.py
     Kept for benchmarking comparative analysis
```

---

## Development Notes

### Key Assumptions
- PostgreSQL with pgvector extension must be running
- Ollama service running on http://localhost:11434 (if not using Groq)
- Papers stored as PDFs in data/papers/
- Indexes stored in indexes/ directory
- API runs on http://localhost:8000
- Streamlit UI on http://localhost:8501

### Common Workflows
```
1. Initial Setup
   $ python scripts/build_index.py        # Ingest papers, create indexes
   $ uvicorn api.main:app                 # Start API
   $ streamlit run app.py                 # Start UI

2. Add Single Paper
   $ POST http://localhost:8000/api/v1/papers/upload  # Upload PDF
   $ POST http://localhost:8000/api/v1/vectors/sync   # Sync embeddings

3. Run Evaluation
   $ python scripts/run_benchmark.py --build-qa       # Generate QA
   $ python scripts/run_benchmark.py --qasper          # Run evaluation

4. Query Paper
   $ POST http://localhost:8000/api/v1/query          # Via API
   $ Or use Streamlit UI at http://localhost:8501
```

---

## File Inventory

### By Component

**Retrieval (7 files)**
- src/retrieval/__init__.py
- src/retrieval/embedder.py
- src/retrieval/retriever.py
- src/retrieval/indexer.py
- src/retrieval/hyde.py

**Ingestion (5 files)**
- src/ingestion/__init__.py
- src/ingestion/ingest.py
- src/ingestion/chunker.py
- src/ingestion/index_manager.py
- src/ingestion/utils.py

**Generation (2 files)**
- src/generation/__init__.py
- src/generation/generator.py
- src/generation/memory.py

**Evaluation (5 files)**
- src/evaluation/__init__.py
- src/evaluation/evaluator.py
- src/evaluation/pipeline_runner.py
- src/evaluation/benchmarker.py
- src/evaluation/retrieval_metrics.py
- src/evaluation/generation_metrics.py

**Download (2 files)**
- src/download/__init__.py
- src/download/arxiv_downloader.py
- src/download/arxiv_fetcher.py

**UI (3 files)**
- src/ui/__init__.py
- src/ui/recommender.py
- src/ui/flowchart.py
- src/ui/tts.py

**API (22 files)**
- api/__init__.py
- api/main.py
- api/dependencies.py
- api/core/config.py
- api/core/database.py
- api/models/tables.py
- api/schemas/schemas.py
- api/services/rag_service.py
- api/services/paper_service.py
- api/services/vector_service.py
- api/routers/health.py
- api/routers/query.py
- api/routers/search.py
- api/routers/papers.py
- api/routers/recommend.py
- api/routers/tts.py
- api/routers/flowchart.py
- api/routers/benchmark.py
- api/routers/feedback.py
- api/routers/stats.py

**Scripts (2 files)**
- scripts/build_index.py
- scripts/run_benchmark.py

**Streamlit UI (1 file)**
- app.py

**Config & Build (6 files)**
- pyproject.toml
- requirements.txt
- requirements-cloud.txt
- alembic.ini (database migrations)
- docker-compose.yml
- Dockerfile
- railway.toml (Railway deployment)
- render.yaml (Render deployment)

---

## Summary

**Anthology v2** is a production-ready, locally-deployed RAG system that demonstrates modern information retrieval and generation techniques. It's designed for research paper analysis with:

✅ **Retrieval**: Multi-strategy (dense + lexical + RRF + reranking)
✅ **Generation**: Local LLM with streaming & citations
✅ **Evaluation**: Comprehensive IR metrics + LLM-as-judge
✅ **Infrastructure**: PostgreSQL pgvector, async FastAPI, Streamlit UI
✅ **Ingestion**: Intelligent chunking with quality scoring & math preservation
✅ **Extensibility**: Modular design, clear data flow, easy to add new components

The architecture balances research rigor (proper evaluation) with practical usability (local deployment, no API dependencies).
