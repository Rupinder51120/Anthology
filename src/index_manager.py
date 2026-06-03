import json
from pathlib import Path
import numpy as np

# Import helper functions from build_index.py (in workspace root)
import sys
sys.path.append(str(Path(__file__).parent.parent))

from build_index import (
    load_checkpoint,
    save_checkpoint,
    filter_chunks,
    preserve_math
)

from src.ingest import load_paper, save_metadata
from src.chunker import chunk_paper, save_chunks
from src.embedder import (
    embed_chunks,
    embed_papers_for_recommendation,
    save_embeddings,
    load_embeddings
)
from src.indexer import (
    build_faiss_index,
    save_faiss_index,
    build_bm25_index,
    save_bm25_index
)


def add_paper(pdf_path: str | Path):
    """
    Ingests, chunks, embeds, and indexes a single PDF paper incrementally.
    Appends to existing chunk/paper metadata and rebuilds FAISS/BM25 indexes.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    checkpoint = load_checkpoint()
    if pdf_path.name in checkpoint.get("processed_files", []):
        print(f"Paper '{pdf_path.name}' is already indexed.")
        return

    print(f"Incrementally adding paper: {pdf_path.name}")

    # Step 1: Ingest & preserve math
    paper = load_paper(pdf_path)
    paper["full_text"] = preserve_math(paper["full_text"])
    for section in paper["sections"]:
        paper["sections"][section] = preserve_math(paper["sections"][section])

    # Step 2: Chunk & filter
    new_raw_chunks = chunk_paper(paper)
    new_chunks, _ = filter_chunks(new_raw_chunks, min_score=0.3)

    if not new_chunks:
        print(f"No high quality chunks extracted from {pdf_path.name}. Index not modified.")
        return

    # Step 3: Append to chunks metadata
    chunks_path = Path("indexes/chunks_metadata.json")
    if chunks_path.exists():
        with open(chunks_path) as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []
    combined_chunks = existing_chunks + new_chunks
    save_chunks(combined_chunks)

    # Step 4: Embed new chunks and append
    new_chunk_embeddings = embed_chunks(new_chunks)
    embeddings_path = Path("indexes/chunk_embeddings.npy")
    if embeddings_path.exists():
        existing_embeddings = load_embeddings(str(embeddings_path))
        combined_embeddings = np.vstack([existing_embeddings, new_chunk_embeddings])
    else:
        combined_embeddings = new_chunk_embeddings
    save_embeddings(combined_embeddings, str(embeddings_path))

    # Step 5: Embed paper for recommendation and append
    new_paper_emb, new_paper_meta = embed_papers_for_recommendation([paper])
    
    paper_embs_path = Path("indexes/paper_embeddings.npy")
    if paper_embs_path.exists():
        existing_paper_embs = load_embeddings(str(paper_embs_path))
        combined_paper_embs = np.vstack([existing_paper_embs, new_paper_emb])
    else:
        combined_paper_embs = new_paper_emb
    save_embeddings(combined_paper_embs, str(paper_embs_path))

    # Append to paper_meta.json
    paper_meta_path = Path("indexes/paper_meta.json")
    if paper_meta_path.exists():
        with open(paper_meta_path) as f:
            existing_paper_meta = json.load(f)
    else:
        existing_paper_meta = []
    combined_paper_meta = existing_paper_meta + new_paper_meta
    with open(paper_meta_path, "w") as f:
        json.dump(combined_paper_meta, f, indent=2)

    # Step 6: Update papers_metadata.json (cumulative list of all ingested paper metadata)
    papers_metadata_path = Path("indexes/papers_metadata.json")
    if papers_metadata_path.exists():
        with open(papers_metadata_path) as f:
            existing_papers_metadata = json.load(f)
    else:
        existing_papers_metadata = []
    
    seen_titles = {p["title"] for p in existing_papers_metadata}
    for pm in new_paper_meta:
        if pm["title"] not in seen_titles:
            existing_papers_metadata.append(pm)
            
    with open(papers_metadata_path, "w") as f:
        json.dump(existing_papers_metadata, f, indent=2)

    # Step 7: Rebuild vector and BM25 index
    print("Rebuilding search indexes...")
    faiss_index = build_faiss_index(combined_embeddings)
    save_faiss_index(faiss_index)

    bm25_index = build_bm25_index(combined_chunks)
    save_bm25_index(bm25_index)

    # Step 8: Update checkpoint
    checkpoint["processed_files"].append(pdf_path.name)
    save_checkpoint(checkpoint)
    print(f"Incremental indexing complete for: {pdf_path.name}")


def add_new_papers(papers_dir: str | Path):
    """
    Scans papers_dir for all PDFs, checks the checkpoint, and index all new papers
    in a single batch for better performance (rebuilds indexes only once).
    """
    papers_path = Path(papers_dir)
    pdf_files = list(papers_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {papers_dir}")
        return

    checkpoint = load_checkpoint()
    already_done = set(checkpoint.get("processed_files", []))
    to_process = [f for f in pdf_files if f.name not in already_done]

    if not to_process:
        print("No new papers to index.")
        return

    print(f"Processing {len(to_process)} new papers...")

    new_papers = []
    new_chunks = []
    processed_filenames = []

    for pdf_path in to_process:
        try:
            print(f"Ingesting: {pdf_path.name}")
            paper = load_paper(pdf_path)
            paper["full_text"] = preserve_math(paper["full_text"])
            for section in paper["sections"]:
                paper["sections"][section] = preserve_math(paper["sections"][section])
            new_papers.append(paper)

            raw_chunks = chunk_paper(paper)
            filtered, _ = filter_chunks(raw_chunks, min_score=0.3)
            new_chunks.extend(filtered)
            
            processed_filenames.append(pdf_path.name)
        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")

    if not new_papers:
        print("No papers were successfully ingested.")
        return

    # Update chunks metadata file
    chunks_path = Path("indexes/chunks_metadata.json")
    if chunks_path.exists():
        with open(chunks_path) as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []
    combined_chunks = existing_chunks + new_chunks
    save_chunks(combined_chunks)

    # Embed new chunks and stack
    new_chunk_embeddings = embed_chunks(new_chunks)
    embeddings_path = Path("indexes/chunk_embeddings.npy")
    if embeddings_path.exists():
        existing_embeddings = load_embeddings(str(embeddings_path))
        combined_embeddings = np.vstack([existing_embeddings, new_chunk_embeddings])
    else:
        combined_embeddings = new_chunk_embeddings
    save_embeddings(combined_embeddings, str(embeddings_path))

    # Embed papers for recommendation
    new_paper_embs, new_paper_meta = embed_papers_for_recommendation(new_papers)
    
    paper_embs_path = Path("indexes/paper_embeddings.npy")
    if paper_embs_path.exists():
        existing_paper_embs = load_embeddings(str(paper_embs_path))
        combined_paper_embs = np.vstack([existing_paper_embs, new_paper_embs])
    else:
        combined_paper_embs = new_paper_embs
    save_embeddings(combined_paper_embs, str(paper_embs_path))

    # Save paper meta
    paper_meta_path = Path("indexes/paper_meta.json")
    if paper_meta_path.exists():
        with open(paper_meta_path) as f:
            existing_paper_meta = json.load(f)
    else:
        existing_paper_meta = []
    combined_paper_meta = existing_paper_meta + new_paper_meta
    with open(paper_meta_path, "w") as f:
        json.dump(combined_paper_meta, f, indent=2)

    # Save papers metadata
    papers_metadata_path = Path("indexes/papers_metadata.json")
    if papers_metadata_path.exists():
        with open(papers_metadata_path) as f:
            existing_papers_metadata = json.load(f)
    else:
        existing_papers_metadata = []
        
    seen_titles = {p["title"] for p in existing_papers_metadata}
    for pm in new_paper_meta:
        if pm["title"] not in seen_titles:
            existing_papers_metadata.append(pm)
            
    with open(papers_metadata_path, "w") as f:
        json.dump(existing_papers_metadata, f, indent=2)

    # Rebuild search indexes
    print("Rebuilding search indexes...")
    faiss_index = build_faiss_index(combined_embeddings)
    save_faiss_index(faiss_index)

    bm25_index = build_bm25_index(combined_chunks)
    save_bm25_index(bm25_index)

    # Update checkpoint
    checkpoint["processed_files"].extend(processed_filenames)
    save_checkpoint(checkpoint)
    print(f"Batch indexing complete. Indexed {len(new_papers)} papers.")


def full_rebuild(papers_dir: str | Path):
    """
    Clears checkpoint and rebuilds search indexes from scratch using all files in papers_dir.
    """
    import build_index
    build_index.PAPERS_DIR = str(papers_dir)
    print(f"Triggering full rebuild using papers in: {papers_dir}")
    build_index.main(force_rebuild=True)

