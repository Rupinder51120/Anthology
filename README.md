# Anthology

> The story behind every discovery.

Anthology is a local-first Retrieval-Augmented Generation (RAG) system for academic research papers. It enables users to ingest PDF collections, retrieve relevant evidence using hybrid search, and generate citation-grounded answers through a conversational interface.

---

## Overview

Anthology combines semantic retrieval, lexical retrieval, reranking, and local language models to create an end-to-end research assistant that runs entirely on your own infrastructure.

The system is designed for:

- Research paper question answering
- Literature exploration
- Evidence-grounded responses
- Private, local-first workflows

---

## Features

- PDF ingestion and indexing
- Hybrid retrieval (FAISS + BM25)
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- HyDE query expansion
- Citation-grounded answers
- Conversational memory
- Local LLM inference via Ollama
- Paper recommendations
- Benchmarking and retrieval evaluation

---

## Architecture

```text
PDFs
 │
 ▼
PyMuPDF
 │
 ▼
Chunking + Metadata
 │
 ├── FAISS (Dense Retrieval)
 │
 └── BM25 (Lexical Retrieval)
        │
        ▼
Reciprocal Rank Fusion
        │
        ▼
Cross Encoder Reranker
        │
        ▼
Context Builder
        │
        ▼
Ollama (Qwen)
        │
        ▼
Citation-Grounded Response
```

---

## Tech Stack

### Backend

- Python
- FastAPI
- PostgreSQL
- Redis
- SQLAlchemy
- Alembic

### Retrieval

- FAISS
- BM25
- Sentence Transformers
- Cross Encoder Reranking
- HyDE

### AI

- Ollama
- Qwen
- Hugging Face Transformers

### Infrastructure

- Docker
- Docker Compose
- Railway / Render

### Frontend

- Streamlit

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
└── docker-compose.yml
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

### Start Services

```bash
docker compose up -d
```

### Run Application

```bash
streamlit run app.py
```

---

## Example Workflow

1. Add research papers to the corpus
2. Build or update indexes
3. Ask questions in natural language
4. Retrieve supporting evidence
5. Generate citation-grounded answers

---

## Future Improvements

- Automated testing
- CI/CD pipelines
- Retrieval observability
- Hallucination guardrails
- Advanced evaluation metrics

---

## License

MIT License
