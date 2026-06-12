# Anthology

### The Story Behind Every Discovery

Anthology is an AI Research Intelligence System designed to help researchers explore, retrieve, and understand academic literature through citation-grounded question answering.

Built as an end-to-end Retrieval-Augmented Generation (RAG) platform, Anthology combines research paper ingestion, retrieval, evaluation, and production deployment into a unified research assistant capable of searching and reasoning across a corpus of scientific papers.

The project began as a research-focused RAG system and evolved into a production-ready platform featuring benchmarking, evaluation, API deployment, and an upcoming multimodal pipeline for understanding figures, tables, and diagrams within research papers.

---

## Live Deployment

**API:** `https://anthology-api.onrender.com`

**Swagger Documentation:** `https://anthology-api.onrender.com/docs`

---

# Key Features

## Research Paper Intelligence

* Research paper ingestion and indexing
* Citation-grounded question answering
* Academic search and retrieval
* Paper metadata management
* Research recommendation workflows
* ArXiv paper integration

---

## Retrieval System

### Current Production Retrieval

* PostgreSQL Full-Text Search (FTS)
* Reciprocal Rank Fusion (RRF)
* Context-aware retrieval pipeline
* Citation tracking and source attribution

### Historical Retrieval Experiments

Anthology includes a benchmarking framework that compares multiple retrieval architectures:

* BM25
* FAISS Dense Retrieval
* Hybrid Retrieval
* Cross-Encoder Reranking
* HyDE Query Expansion
* Reciprocal Rank Fusion

---

## Generation

* Citation-grounded answer generation
* Structured research responses
* Multi-document reasoning
* Local LLM support via Ollama
* API-based LLM integration
* Conversation memory support

---

## Evaluation Framework

Anthology includes a dedicated retrieval and answer-quality evaluation framework.

### Retrieval Metrics

* Hit@1
* Hit@3
* Hit@5
* Mean Reciprocal Rank (MRR)
* nDCG@5

### Answer Quality Metrics

* Faithfulness
* Relevance
* Completeness

### Benchmarking

* Retrieval configuration comparison
* Automated QA benchmark generation
* QASPER-style evaluation datasets
* End-to-end pipeline benchmarking

---

# Architecture

```text
Research Papers (PDFs)
          │
          ▼
     Ingestion Layer
          │
          ▼
  Section-Aware Chunking
          │
          ▼
      PostgreSQL
          │
          ▼
     Retrieval Layer
          │
          ▼
 Context Construction
          │
          ▼
     LLM Generation
          │
          ▼
 Citation Grounding
          │
          ▼
 Research Answer
```

---

# System Components

```text
Anthology
│
├── Research Paper Ingestion
├── Retrieval Engine
├── Generation Engine
├── Evaluation Framework
├── Recommendation System
├── REST API Layer
├── Database Layer
└── Deployment Infrastructure
```

---

# Technology Stack

## AI & Retrieval

* Retrieval-Augmented Generation (RAG)
* PostgreSQL Full-Text Search
* FAISS (Benchmarking)
* BM25 (Benchmarking)
* Reciprocal Rank Fusion
* Cross-Encoder Reranking
* HyDE Query Expansion
* Ollama
* Hugging Face

---

## Backend

* Python
* FastAPI
* PostgreSQL
* SQLAlchemy
* Alembic
* AsyncIO

---

## Frontend

* Streamlit

---

## Infrastructure

* Docker
* Docker Compose
* Render
* GitHub

---

## Data Processing

* PyMuPDF
* ArXiv API
* Pandas
* NumPy

---

# Project Structure

```text
Anthology/
│
├── api/
│   ├── routers/
│   ├── services/
│   ├── models/
│   └── core/
│
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   ├── recommendation/
│   └── ui/
│
├── scripts/
├── data/
├── indexes/
├── alembic/
├── multimodal/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Evaluation Results

Historical benchmark results across a 100-question evaluation set:

| Configuration   | Hit@5    | MRR        | Mean Score |
| --------------- | -------- | ---------- | ---------- |
| BM25            | 0.94     | 0.8378     | 3.7767     |
| FAISS           | 0.93     | 0.8712     | 3.8033     |
| Hybrid          | **0.96** | **0.8925** | **3.8800** |
| Hybrid + Rerank | 0.95     | 0.8562     | 3.7700     |
| Hybrid + HyDE   | 0.94     | 0.8443     | 3.7633     |

Best-performing historical configuration:

```text
Hit@5  : 96%
MRR    : 0.8925
nDCG@5 : 0.8923
```

---

# Local Development

## Clone Repository

```bash
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment

```bash
cp .env.example .env
```

## Run Database

```bash
docker compose up -d
```

## Start API

```bash
uvicorn api.main:app --reload
```

## Open Documentation

```text
http://localhost:8000/docs
```

---

# Current Status

### Production

* Deployed FastAPI backend
* PostgreSQL database
* Public REST API
* Swagger documentation
* Research paper corpus
* Citation-grounded QA

### Research Infrastructure

* Retrieval benchmarking
* Evaluation framework
* QASPER-style benchmarks
* Retrieval experimentation

### In Progress

* Multimodal document understanding
* Figure analysis
* Table understanding
* Diagram interpretation
* Chart reasoning
* Multimodal retrieval

---

# Resume Highlights

Anthology demonstrates experience with:

* Retrieval-Augmented Generation (RAG)
* Information Retrieval Systems
* FastAPI Backend Development
* PostgreSQL Database Design
* Production Deployment
* Evaluation & Benchmarking
* Research Data Processing
* API Development
* Asynchronous Python
* AI System Architecture

---

# Future Roadmap

### Short-Term

* Multimodal PDF understanding
* Figure retrieval
* Table retrieval
* Diagram explanation
* Enhanced answer quality

### Long-Term

* Research workspace
* Collaboration features
* Citation graph exploration
* Advanced evaluation dashboards
* Production-scale multimodal deployment

---

# License

MIT License
