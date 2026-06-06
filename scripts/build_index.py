"""
Advanced index builder with:
- Parallel PDF ingestion
- Chunk quality scoring + filtering
- Math/LaTeX preservation
- Progress tracking
- Resume from checkpoint (skip already processed papers)
- Build report with stats
"""
import json
import time
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

from src.ingestion.ingest import load_paper, save_metadata
from src.ingestion.chunker import chunk_paper, save_chunks
from src.retrieval.embedder import embed_chunks, embed_papers_for_recommendation, save_embeddings
from src.retrieval.indexer import build_faiss_index, save_faiss_index, build_bm25_index, save_bm25_index


# ─── paths ────────────────────────────────────────────────────

PAPERS_DIR      = "data/papers"
INDEXES_DIR     = "indexes"
CHECKPOINT_PATH = "indexes/build_checkpoint.json"
REPORT_PATH     = "indexes/build_report.json"


# ─── chunk quality scorer ─────────────────────────────────────

def score_chunk(chunk: dict) -> float:
    text  = chunk["text"].strip()
    score = 0.0

    length = len(text)
    if length < 80:
        return 0.0
    elif length < 150:
        score += 0.1
    elif length <= 500:
        score += 0.4
    else:
        score += 0.3

    sentences = [s for s in re.split(r'[.!?]', text) if len(s.strip()) > 10]
    if len(sentences) >= 2:
        score += 0.2
    if len(sentences) >= 4:
        score += 0.1

    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    if len(numbers) >= 2:
        score += 0.1

    math_patterns = [r'\$.*?\$', r'\\[a-zA-Z]+', r'[αβγδεζηθλμπσφψω]',
                     r'=\s*\d', r'\^{', r'_{', r'\\frac', r'\\sum', r'\\int']
    if any(re.search(p, text) for p in math_patterns):
        score += 0.15

    stopwords = {'the', 'a', 'an', 'is', 'it', 'in', 'of', 'to', 'and',
                 'or', 'for', 'on', 'at', 'by', 'be', 'as', 'this', 'that'}
    words = text.lower().split()
    if words:
        stopword_ratio = sum(1 for w in words if w in stopwords) / len(words)
        if stopword_ratio > 0.6:
            score -= 0.2

    return max(0.0, min(1.0, score))


def filter_chunks(chunks: list[dict], min_score: float = 0.3) -> tuple[list[dict], dict]:
    scored  = [(chunk, score_chunk(chunk)) for chunk in chunks]
    kept    = [c for c, s in scored if s >= min_score]
    removed = [c for c, s in scored if s < min_score]

    for chunk, score in scored:
        chunk["metadata"]["quality_score"] = round(score, 3)

    stats = {
        "total":        len(chunks),
        "kept":         len(kept),
        "removed":      len(removed),
        "removal_rate": f"{len(removed)/len(chunks)*100:.1f}%" if chunks else "0%"
    }
    return kept, stats


# ─── math/LaTeX preservation ──────────────────────────────────

def preserve_math(text: str) -> str:
    text = re.sub(r'\$\$([^$]+)\$\$', r'[MATH_BLOCK]\1[/MATH_BLOCK]', text)
    text = re.sub(r'\$([^$]+)\$',     r'[MATH]\1[/MATH]', text)
    text = re.sub(r'(\\[a-zA-Z]+\{[^}]*\})', r'[MATH]\1[/MATH]', text)
    return text


# ─── checkpoint ───────────────────────────────────────────────

def load_checkpoint() -> dict:
    if Path(CHECKPOINT_PATH).exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        if "processed_files" not in data:
            data["processed_files"] = []
        return data
    return {"processed_files": [], "last_run": None}


def save_checkpoint(checkpoint: dict):
    Path(INDEXES_DIR).mkdir(exist_ok=True)
    checkpoint["last_run"] = datetime.now().isoformat()
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ─── parallel ingestion ───────────────────────────────────────

def load_paper_safe(pdf_path) -> dict | None:
    try:
        paper = load_paper(pdf_path)
        paper["full_text"] = preserve_math(paper["full_text"])
        for section in paper["sections"]:
            paper["sections"][section] = preserve_math(paper["sections"][section])
        return paper
    except Exception as e:
        print(f"  Failed: {pdf_path.name} — {e}")
        return None


def load_papers_parallel(
    papers_dir: str,
    checkpoint: dict,
    max_workers: int = 4
) -> list[dict]:

    papers_path = Path(papers_dir)
    pdf_files   = list(papers_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {papers_dir}")
        return []

    # ensure key exists
    if "processed_files" not in checkpoint:
        checkpoint["processed_files"] = []

    already_done = set(checkpoint["processed_files"])
    to_process   = [f for f in pdf_files if f.name not in already_done]
    skipped      = len(pdf_files) - len(to_process)

    if skipped:
        print(f"Checkpoint: skipping {skipped} already-processed papers")

    print(f"Processing: {len(to_process)} new papers with {max_workers} workers\n")

    if not to_process:
        return []

    results = []
    failed  = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(load_paper_safe, f): f
            for f in to_process
        }

        for i, future in enumerate(as_completed(future_to_file), 1):
            pdf_file = future_to_file[future]
            try:
                paper = future.result()
            except Exception as e:
                print(f"  [{i}/{len(to_process)}] ✗ {pdf_file.name[:55]} — {e}")
                failed.append(pdf_file.name)
                continue

            if paper:
                results.append(paper)
                checkpoint["processed_files"].append(pdf_file.name)
                print(f"  [{i}/{len(to_process)}] ✓ {pdf_file.name[:55]}")
            else:
                failed.append(pdf_file.name)
                print(f"  [{i}/{len(to_process)}] ✗ {pdf_file.name[:55]}")

    if failed:
        print(f"\nFailed: {len(failed)} papers — {failed}")

    print(f"\nLoaded {len(results)} papers successfully")
    return results


# ─── progress printer ─────────────────────────────────────────

def print_step(step: int, total: int, title: str):
    bar = "█" * step + "░" * (total - step)
    print(f"\n[{bar}] STEP {step}/{total}: {title}")
    print("─" * 50)


# ─── main ─────────────────────────────────────────────────────

def main(
    force_rebuild:     bool  = False,
    min_chunk_quality: float = 0.3,
    max_workers:       int   = 4
):
    start_time = time.time()
    Path(INDEXES_DIR).mkdir(exist_ok=True)

    build_stats = {
        "started_at": datetime.now().isoformat(),
        "config": {
            "force_rebuild":     force_rebuild,
            "min_chunk_quality": min_chunk_quality,
            "max_workers":       max_workers
        }
    }

    # checkpoint
    checkpoint = {"processed_files": [], "last_run": None} if force_rebuild else load_checkpoint()
    if force_rebuild:
        print("Force rebuild — ignoring checkpoint")

    # ══ STEP 1 ══════════════════════════════════════════════
    print_step(1, 6, "Parallel PDF ingestion + math preservation")

    papers = load_papers_parallel(PAPERS_DIR, checkpoint, max_workers=max_workers)

    if not papers:
        print("No new papers to process. Use --force to rebuild everything.")
        return

    save_metadata(papers)
    save_checkpoint(checkpoint)
    build_stats["papers_loaded"] = len(papers)

    # ══ STEP 2 ══════════════════════════════════════════════
    print_step(2, 6, "Section-aware chunking")

    raw_chunks = []
    for paper in papers:
        paper_chunks = chunk_paper(paper)
        raw_chunks.extend(paper_chunks)
        print(f"  {paper['metadata']['filename'][:45]}: {len(paper_chunks)} chunks")

    print(f"\nRaw chunks: {len(raw_chunks)}")
    print(f"Applying quality filter (min score: {min_chunk_quality})...")

    filtered_chunks, quality_stats = filter_chunks(raw_chunks, min_score=min_chunk_quality)

    print(f"  Total:   {quality_stats['total']}")
    print(f"  Kept:    {quality_stats['kept']}")
    print(f"  Removed: {quality_stats['removed']} ({quality_stats['removal_rate']})")

    save_chunks(filtered_chunks)
    build_stats["chunking"] = quality_stats

    # ══ STEP 3 ══════════════════════════════════════════════
    print_step(3, 6, "Embedding chunks with SPECTER2")

    chunk_embeddings = embed_chunks(filtered_chunks)
    save_embeddings(chunk_embeddings, f"{INDEXES_DIR}/chunk_embeddings.npy")
    build_stats["embedding_shape"] = list(chunk_embeddings.shape)

    # ══ STEP 4 ══════════════════════════════════════════════
    print_step(4, 6, "Embedding papers for recommendation engine")

    paper_embeddings, paper_meta = embed_papers_for_recommendation(papers)
    save_embeddings(paper_embeddings, f"{INDEXES_DIR}/paper_embeddings.npy")

    with open(f"{INDEXES_DIR}/paper_meta.json", "w") as f:
        json.dump(paper_meta, f, indent=2)

    build_stats["paper_embeddings"] = len(paper_meta)

    # ══ STEP 5 ══════════════════════════════════════════════
    print_step(5, 6, "Building FAISS index")

    faiss_index = build_faiss_index(chunk_embeddings)
    save_faiss_index(faiss_index)
    build_stats["faiss_vectors"] = int(faiss_index.ntotal)

    # ══ STEP 6 ══════════════════════════════════════════════
    print_step(6, 6, "Building BM25 index")

    bm25_index = build_bm25_index(filtered_chunks)
    save_bm25_index(bm25_index)

    # final report
    elapsed = round(time.time() - start_time, 1)
    build_stats["finished_at"]     = datetime.now().isoformat()
    build_stats["elapsed_seconds"] = elapsed

    with open(REPORT_PATH, "w") as f:
        json.dump(build_stats, f, indent=2)

    print(f"\n{'═'*50}")
    print(f"BUILD COMPLETE in {elapsed}s")
    print(f"{'═'*50}")
    print(f"Papers loaded:    {build_stats['papers_loaded']}")
    print(f"Chunks kept:      {quality_stats['kept']} / {quality_stats['total']}")
    print(f"Removed (low Q):  {quality_stats['removed']} ({quality_stats['removal_rate']})")
    print(f"Embedding dim:    {chunk_embeddings.shape[1]}")
    print(f"FAISS vectors:    {build_stats['faiss_vectors']}")
    print(f"Report saved:     {REPORT_PATH}")
    print(f"{'═'*50}")
    print(f"\nNext: streamlit run app.py")


# ─── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build RAG indexes from research papers"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force full rebuild, ignore checkpoint"
    )
    parser.add_argument(
        "--quality", "-q",
        type=float,
        default=0.3,
        help="Minimum chunk quality score 0.0–1.0 (default: 0.3)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Parallel workers for PDF loading (default: 4)"
    )

    args = parser.parse_args()
    main(
        force_rebuild=args.force,
        min_chunk_quality=args.quality,
        max_workers=args.workers
    )