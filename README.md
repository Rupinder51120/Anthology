# Anthology

> **The Story Behind Every Discovery**

Anthology is a local-first Retrieval-Augmented Generation (RAG) system for academic literature. It enables users to ingest research papers, retrieve relevant evidence through hybrid search, and generate citation-grounded answers using local language models.

Built to explore modern information retrieval techniques, Anthology combines dense retrieval, lexical retrieval, reranking, query expansion, and evaluation into a single end-to-end research assistant.

---

## Overview

Researchers often work with hundreds of papers spread across multiple domains, making knowledge retrieval difficult and time-consuming.

Anthology addresses this problem by:

* Ingesting and indexing research papers
* Retrieving relevant passages using hybrid search
* Generating answers grounded in retrieved evidence
* Providing citations for transparency
* Evaluating retrieval quality using IR metrics

The entire system runs locally, allowing private and reproducible research workflows.

---

## Features

### Retrieval

* Dense semantic retrieval with FAISS
* Lexical retrieval with BM25
* Hybrid retrieval via Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* HyDE query expansion

### Generation

* Local LLM inference through Ollama
* Streaming responses
* Citation-grounded answers
* Multi-turn conversation memory

### Research Tools

* Research paper recommendations
* ArXiv paper ingestion
* Library management
* Retrieval benchmarking

### Evaluation

* Hit@K
* Mean Reciprocal Rank (MRR)
* nDCG@K
* Retrieval configuration comparison

---

## Architecture

```text
Research Papers (PDFs)
          │
          ▼
      PyMuPDF
          │
          ▼
 Section-Aware Chunking
          │
 ┌────────┴────────┐
 ▼                 ▼
FAISS            BM25
(Dense)        (Lexical)
 └────────┬────────┘
          ▼
Reciprocal Rank Fusion
          ▼
 Cross-Encoder Reranker
          ▼
      HyDE Expansion
          ▼
 Context Construction
          ▼
 Ollama
          ▼
 Citation-Grounded Answer
```

---

## Retrieval Pipeline

```text
User Query
    │
    ▼
Intent Detection
    │
    ▼
Hybrid Retrieval
(FAISS + BM25)
    │
    ▼
Reciprocal Rank Fusion
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Context Construction
    │
    ▼
LLM Generation
    │
    ▼
Citation Formatting
```

---

## Tech Stack

### AI & Retrieval

* FAISS
* BM25
* Sentence Transformers
* Cross-Encoder Reranking
* HyDE
* Ollama
* Hugging Face Transformers

### Backend

* Python
* FastAPI
* PostgreSQL
* Redis
* SQLAlchemy
* Alembic

### Frontend

* Streamlit

### Infrastructure

* Docker
* Docker Compose
* Railway / Render

### Data Processing

* PyMuPDF
* ArXiv API
* Pandas

---

## Project Structure

```text
Anthology/
│
├── app.py
├── api/
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   ├── download/
│   └── ui/
│
├── scripts/
├── data/
├── alembic/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Local Setup

### Clone Repository

```bash
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
```

### Start Infrastructure

```bash
docker compose up -d
```

### Run Application

```bash
streamlit run app.py
```

---

## Evaluation

Anthology includes a retrieval evaluation framework for comparing different retrieval configurations and measuring search quality across academic corpora.

Metrics include:

* Hit@1
* Hit@3
* Hit@5
* Mean Reciprocal Rank (MRR)
* nDCG@5

---

## Roadmap

* Automated testing
* GitHub Actions CI/CD
* Retrieval observability
* LLM guardrails
* Expanded evaluation metrics
* Production deployment

---

## License

MIT License
