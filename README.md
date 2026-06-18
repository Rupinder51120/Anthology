# Anthology

## AI Research Intelligence System

Anthology is a Retrieval-Augmented Generation (RAG) platform for exploring and querying scientific literature.

The system ingests research papers, indexes their content, retrieves relevant evidence through a hybrid retrieval pipeline, and generates citation-grounded responses. It is designed as a full-stack application with a FastAPI backend, PostgreSQL/pgvector storage, and a React frontend.

---

## Features

* Research paper ingestion and indexing
* Section-aware document chunking
* Hybrid retrieval using:

  * PostgreSQL Full-Text Search (FTS)
  * pgvector semantic search
* SPECTER2 scientific embeddings
* Reciprocal Rank Fusion (RRF)
* Cohere reranking
* Citation-grounded answer generation
* Retrieval benchmarking and evaluation
* Langfuse observability and tracing

---

## Architecture

```text
PDF
 │
 ▼
Document Parsing
 │
 ▼
Chunking
 │
 ▼
Embedding Generation
 │
 ▼
PostgreSQL + pgvector
 │
 ▼
Hybrid Retrieval
(FTS + Vector Search)
 │
 ▼
RRF Fusion
 │
 ▼
Reranking
 │
 ▼
LLM Generation
 │
 ▼
Answer + Citations
```

---

## Tech Stack

### Backend

* FastAPI
* SQLAlchemy
* Alembic
* Pydantic

### Database

* PostgreSQL
* pgvector
* PostgreSQL Full-Text Search

### Retrieval

* SPECTER2
* Cohere Rerank
* Reciprocal Rank Fusion (RRF)

### LLMs

* Groq
* Ollama

### Frontend

* React
* TypeScript
* Vite

### Observability

* Langfuse

---

## Project Structure

```text
Anthology/
│
├── api/
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
│
├── frontend/
├── scripts/
├── alembic/
├── docker/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology
```

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
DATABASE_URL=
GROQ_API_KEY=
COHERE_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

### Start Services

```bash
docker compose up -d
alembic upgrade head
```

### Run the API

```bash
uvicorn api.main:app --reload
```

API Documentation:

```text
http://localhost:8000/docs
```


## License

MIT License
