# Research Paper RAG System

A production-grade Retrieval-Augmented Generation system for querying academic papers, built entirely with local models — no API keys, no rate limits, fully reproducible.

## What it does

- Ingests research papers (PDF) and chunks them with section-aware splitting
- Hybrid retrieval: FAISS semantic search + BM25 lexical search with Reciprocal Rank Fusion
- Cross-encoder reranking and HyDE query expansion
- Local LLM answer generation via Ollama (qwen2.5:7b)
- Streamlit chat UI with streaming responses, citations, and paper recommendations
- Full evaluation pipeline with bias-audited benchmarks

## Benchmark Results

Evaluated on a 100-question bias-audited benchmark (abstract-anchored QA generation):

| Config | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| BM25 baseline | 0.77 | 0.90 | 0.94 | 0.838 | 0.854 |
| FAISS only | 0.84 | 0.89 | 0.93 | 0.871 | 0.883 |
| Hybrid (best) | 0.84 | 0.95 | 0.96 | 0.893 | 0.892 |
| Hybrid + rerank | 0.78 | 0.93 | 0.95 | 0.856 | 0.869 |
| Hybrid + HyDE | 0.79 | 0.89 | 0.94 | 0.844 | 0.843 |

Key finding: Chunk-derived QA benchmarks inflate BM25 scores via lexical leakage. After switching to abstract-anchored generation, Hybrid retrieval outperforms BM25 across all metrics.

## Stack

| Component | Technology |
|---|---|
| PDF parsing | PyMuPDF |
| Embeddings | BAAI/bge-large-en-v1.5 (1024-dim) |
| Vector search | FAISS (IndexFlatIP) |
| Lexical search | BM25Okapi |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Ollama qwen2.5:7b |
| UI | Streamlit |

## Setup

```bash
git clone https://github.com/Rupinder51120/RAG.git
cd RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
make index
make app
```

## Project Structure

```
RAG/
├── app.py                      # Streamlit UI
├── scripts/
│   ├── build_index.py          # PDF ingestion + index building
│   └── run_benchmark.py        # Evaluation pipeline
├── src/
│   ├── ingestion/              # PDF parsing, chunking
│   ├── retrieval/              # FAISS, BM25, embedder, HyDE
│   ├── generation/             # LLM answer generation
│   ├── evaluation/             # Benchmarking + metrics
│   ├── download/               # ArXiv paper downloader
│   └── ui/                     # Streamlit helpers
└── data/
    ├── papers/                 # PDF files
    └── download_registry.json  # Paper metadata
```

## Evaluation Methodology

Standard RAG benchmarks suffer from lexical bias: when QA pairs are generated from chunk text, questions inherit rare tokens that BM25 matches trivially, inflating BM25 scores by ~15%.

This project addresses this by:
1. Generating questions from paper abstracts only (QASPER-style)
2. Filtering questions with content-word Jaccard overlap > 0.35 against source chunks
3. Tracking generation_source per QA pair for bias diagnostics
