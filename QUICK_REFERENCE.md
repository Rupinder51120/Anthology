# Anthology v2 - Quick Reference Guide

## 🗺️ Component Map

| Component | Location | Purpose | Key Files |
|-----------|----------|---------|-----------|
| **Retrieval** | `src/retrieval/` | Dense + lexical search, RRF, reranking | embedder.py, retriever.py, indexer.py, hyde.py |
| **Ingestion** | `src/ingestion/` | PDF→chunks→embeddings→index | ingest.py, chunker.py, index_manager.py, utils.py |
| **Generation** | `src/generation/` | LLM inference, citations, memory | generator.py, memory.py |
| **Evaluation** | `src/evaluation/` | Metrics, benchmarking, LLM judge | retrieval_metrics.py, generation_metrics.py, pipeline_runner.py, benchmarker.py |
| **Download** | `src/download/` | ArXiv fetcher, paper discovery | arxiv_downloader.py |
| **UI** | `src/ui/` | Recommendations, flowcharts, TTS | recommender.py, flowchart.py, tts.py |
| **API** | `api/` | FastAPI endpoints, ORM, services | main.py, routers/*.py, services/*.py, models/tables.py |
| **Streamlit** | `app.py` | Web UI for querying, recommendations | — |
| **Scripts** | `scripts/` | CLI tools: build, benchmark | build_index.py, run_benchmark.py |

---

## 🔄 Data Flow at a Glance

### Ingestion
```
PDF → PyMuPDF → Sections → Chunks (1400 chars) 
→ Quality Score → Type Classify → BGE Embed (384-dim) 
→ NumpyIndex + BM25 + PostgreSQL
```

### Retrieval
```
Query → Embed → pgvector (top-15) + FTS (top-15) 
→ RRF Fusion (top-10) → Cross-Encoder (top-5) → Output
```

### Generation
```
Chunks → Format Context → Ollama/Groq → Format Citations → Response
```

### Evaluation
```
QA Dataset → Retrieve → Generate → Retrieval Metrics + Generation Metrics → Report
```

---

## 🚀 Quick Start

### Build Index from Scratch
```bash
python scripts/build_index.py
# Creates: indexes/chunks_metadata.json, embeddings.npy, bm25_index.pkl
```

### Start API
```bash
cd api
uvicorn main:app --reload
# http://localhost:8000/docs (Swagger UI)
```

### Start Streamlit UI
```bash
streamlit run app.py
# http://localhost:8501
```

### Run Benchmark
```bash
python scripts/run_benchmark.py --qasper
# Compares retrieval configs, computes metrics
```

---

## 📋 Key Configuration

| Setting | Location | Default | Purpose |
|---------|----------|---------|---------|
| Embedding Model | `src/retrieval/embedder.py` | BAAI/bge-large-en-v1.5 | 384-dim embeddings |
| LLM Model | `src/generation/generator.py` | qwen2.5:7b (Ollama) | Local or Groq |
| Chunk Size | `src/ingestion/chunker.py` | 1400 / 1800 (math) | Chars per chunk |
| RRF Parameter | `src/retrieval/retriever.py` | 60 | RRF_K value |
| Min Chunk Score | `scripts/build_index.py` | 0.3 | Quality threshold |
| Database URL | `api/core/config.py` | postgresql+asyncpg://localhost/anthology | PostgreSQL |
| Ollama URL | `src/retrieval/hyde.py` | http://localhost:11434 | Local LLM server |

---

## 🔗 Dependency Graph

```
User Query
  ↓
retrieve(query, top_k, db)
  ├─ embed_texts() [embedder.py]
  ├─ pgvector_search() [retriever.py]
  ├─ postgres_fts_search() [retriever.py]
  ├─ rrf_fuse() [retriever.py]
  └─ rerank() [retriever.py]
  ↓
generate_answer(question, chunks)
  ├─ format_context() [generator.py]
  ├─ Ollama API call
  └─ format_citations() [generator.py]
  ↓
Response (answer + citations)
```

---

## 📊 Performance Metrics

| Operation | Time | Scale |
|-----------|------|-------|
| Embedding chunk | 5-10ms | batch_size=32 |
| pgvector search | <10ms | 2,600 chunks |
| FTS search | <10ms | 2,600 chunks |
| RRF fusion | <1ms | combining 2 lists |
| Cross-encoder rerank | 50-200ms | 5 chunks |
| **Full pipeline** | **2-10s** | **dominated by LLM** |

---

## 🎯 File Locations

### Indexes (in `indexes/`)
```
chunks_metadata.json          ← Chunk data + metadata
chunk_embeddings.npy          ← 2600 × 384 float32 matrix
paper_embeddings.npy          ← 119 × 384 float32 matrix
faiss_index.bin               ← NumpyIndex (pickle)
bm25_index.pkl                ← BM25 (pickle)
paper_meta.json               ← Paper metadata list
checkpoint.json               ← Ingestion checkpoint
session.json                  ← Conversation memory
qa_dataset.json               ← QA pairs for eval
pipeline_results.json         ← Benchmark results
build_report.json             ← Build statistics
```

### Configuration (in `data/`)
```
papers/                       ← PDF files
download_config.json          ← ArXiv topics, max_papers
download_registry.json        ← Downloaded papers metadata
last_download_report.json     ← Download stats
```

### Database Tables
```
papers       ← Paper metadata (arxiv_id, title, authors, etc.)
queries      ← Query history (question, answer, latency, etc.)
feedback     ← User ratings (1-5, comments)
chunks       ← Chunk + embeddings + pgvector (for retrieval)
```

---

## 🔍 Key Classes & Functions

### Retrieval System
```python
# embedder.py
get_model() → SentenceTransformer
embed_texts(texts) → np.ndarray  # 384-dim
embed_chunks(chunks) → np.ndarray

# retriever.py
async retrieve(query, top_k, db) → list[dict]
async pgvector_search(embedding, top_k, db) → list[dict]
async postgres_fts_search(query, top_k, db) → list[dict]
rrf_fuse(vec_results, fts_results, top_k) → list[dict]
rerank(query, chunks, top_k) → list[dict]

# indexer.py
NumpyIndex(embeddings) 
  .search(query, top_k) → (scores, indices)
```

### Generation System
```python
# generator.py
format_context(chunks) → str  # Numbered format
format_citations(chunks) → list[dict]
generate_answer(question, chunks) → dict
generate_answer_streaming(question, chunks) → generator

# memory.py
ConversationMemory(max_turns=6)
  .add(role, content) → None
  .get() → list[dict]
  .save(path) / load(path)
```

### Evaluation Framework
```python
# retrieval_metrics.py
RetrievalMetrics()
  .recall_at_k(retrieved, gold, k) → float
  .mrr(retrieved, gold) → float
  .ndcg_at_k(retrieved, gold, k) → float

# generation_metrics.py
GenerationEvaluator()
  .evaluate_single(question, answer, chunks, gold_answer) → GenerationScore

# pipeline_runner.py
run_pipeline_on_dataset(qa_path, top_k=5, use_hyde=True) → list[dict]
```

### Ingestion System
```python
# ingest.py
load_paper(pdf_path) → dict  # {metadata, sections, full_text}

# chunker.py
chunk_paper(paper) → list[dict]  # Each with text + metadata
detect_chunk_type(text) → str  # math_heavy, figure_table, etc.

# index_manager.py
add_paper(pdf_path) → None  # Incremental add
full_rebuild() → None  # Full index rebuild
```

---

## 🗣️ API Endpoints

### Query Endpoints
```
POST /api/v1/query
├─ Request: {question, top_k, use_hyde}
└─ Response: {question, answer, citations, latency_ms}

POST /api/v1/query/stream
├─ Request: {question, top_k, use_hyde}
└─ Response: streaming tokens
```

### Search Endpoints
```
POST /api/v1/search
├─ Request: {query, top_k, use_hyde}
└─ Response: {query, results: [SearchResult], total}

POST /api/v1/recommend
├─ Request: {query, top_k}
└─ Response: {query, local: [Paper], arxiv: [Paper]}
```

### Management Endpoints
```
GET /api/v1/papers
└─ Response: {papers: [Paper], total}

POST /api/v1/papers/upload
├─ Request: file (multipart)
└─ Response: {success, message}

POST /api/v1/papers/sync
└─ Response: {success, synced}

POST /api/v1/vectors/sync
└─ Response: {success, inserted, total}
```

---

## 🎓 Metrics Reference

### Retrieval Metrics
- **Recall@k**: Fraction of relevant items in top-k (coverage focus)
- **Hit@k**: Binary: any relevant item in top-k (threshold focus)
- **Precision@k**: Fraction of top-k that are relevant (purity focus)
- **MRR**: 1/rank_of_first_relevant (ranking quality)
- **NDCG@k**: Normalized DCG (balanced quality + coverage)
- **AP**: Area under P-R curve (average across all k)

### Generation Metrics
- **Faithfulness** (0-1): % claims grounded in context
- **Answer Relevancy** (0-1): % of answer addressing question
- **Context Precision** (0-1): % of context chunks contributing to answer
- **Completeness** (0-1): coverage vs gold answer (optional)

---

## 🐛 Debugging Tips

### Query Returns Empty Results
```python
# 1. Check embeddings loaded
chunks = load_embeddings("indexes/chunk_embeddings.npy")
print(f"Shape: {chunks.shape}")  # Should be (2600, 384) or similar

# 2. Check PostgreSQL connectivity
async with get_db() as db:
    result = await db.execute(text("SELECT COUNT(*) FROM chunks"))
    print(result.scalar())  # Should be >0

# 3. Check retrieval pipeline
from src.retrieval.retriever import retrieve
chunks = await retrieve("test query", top_k=5, db=db)
print(len(chunks))  # Should be 5
```

### LLM Returns Hallucinations
```python
# Check system prompt enforcement
# In generator.py SYSTEM_PROMPT must include:
# "ONLY use information from the provided context"
# "Never hallucinate citations or results"

# Check citations are extracted
from src.generation.generator import format_citations
citations = format_citations(chunks)
print(len(citations))  # Should match # of sources
```

### Index Building Stalls
```python
# Check checkpoint
import json
with open("indexes/checkpoint.json") as f:
    checkpoint = json.load(f)
print(f"Processed: {len(checkpoint['processed_files'])}")

# Resume
python scripts/build_index.py
# Will skip already-processed files
```

---

## 🔧 Common Modifications

### Change Embedding Model
```python
# src/retrieval/embedder.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller/faster
# Then rebuild indexes: python scripts/build_index.py
```

### Use Groq Instead of Ollama
```python
# .env
USE_GROQ=true
GROQ_API_KEY=xxx
# No code changes needed
```

### Increase Chunk Size for Long Documents
```python
# src/ingestion/chunker.py
CHUNK_SIZE_DEFAULT = 2000  # was 1400
# Then rebuild indexes
```

### Add Custom Metric
```python
# src/evaluation/retrieval_metrics.py
def my_metric(self, retrieved, gold, k):
    # Your metric logic
    return score

# Use in benchmarker
from src.evaluation.retrieval_metrics import RetrievalMetrics
metrics = RetrievalMetrics()
score = metrics.my_metric(...)
```

---

## 📚 Documentation Files

- **ARCHITECTURE.md**: Complete system design (this)
- **ARCHITECTURE_DIAGRAM.md**: Mermaid diagram of data flow
- **ARCHITECTURE_DECISIONS.md**: Design patterns & tradeoffs
- **README.md**: User-facing overview

---

## 🚨 Known Issues & Workarounds

| Issue | Cause | Workaround |
|-------|-------|-----------|
| FAISS segfault on ARM64 | C++ build system | Use NumpyIndex (default) |
| Ollama slow on first query | Model load time | Pre-warm Ollama before API |
| pgvector extension missing | PostgreSQL setup | `CREATE EXTENSION vector;` |
| Chunks with null bytes | PDF parsing artifact | VectorService cleans them |
| OOM during embeddings | Large batch size | Reduce batch_size in embedder.py |

---

## 📞 Key Contacts / Resources

- **Ollama**: http://localhost:11434 (default)
- **PostgreSQL**: postgresql+asyncpg://anthology:anthology@localhost/anthology
- **Streamlit**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Logs**: Check terminal for uvicorn/streamlit output

---

## 🎯 Next Steps for Exploration

1. **Run build_index.py** to understand ingestion pipeline
2. **Query via API** (`/api/v1/query`) to trace retrieval path
3. **Check indexes/** directory to see produced artifacts
4. **Run benchmark** (`scripts/run_benchmark.py --qasper`) to evaluate
5. **Read ARCHITECTURE_DECISIONS.md** for design philosophy
6. **Explore code with IDE** using provided file map

---

## 📊 System at a Glance

```
┌─ Hybrid Retrieval ────────────┐
│  Dense (pgvector) + Lexical (FTS) + RRF + Reranking
│  Top-5 chunks per query
│
├─ Local LLM Generation ────────┐
│  Ollama (qwen2.5:7b) + Groq fallback
│  Structured output with citations
│
├─ Comprehensive Evaluation ────┐
│  Retrieval + Generation metrics
│  Benchmarking framework
│
├─ Async FastAPI ───────────────┐
│  Concurrent queries
│  PostgreSQL + pgvector DB
│
└─ Streamlit UI ────────────────┐
   Query, recommendations, paper upload
```

---

**Last Updated**: 2026-06-11
**System Version**: Anthology v2
**Active Branch**: anthology-v2 (pgvector-migration)
