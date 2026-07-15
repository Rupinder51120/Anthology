import asyncio
import json
import random
import re
from pathlib import Path
from collections import defaultdict

import requests
from sqlalchemy import select
from api.core.database import AsyncSessionLocal
from api.models.tables import Chunk, Paper
from api.core.models import OLLAMA_CHAT_MODEL

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = OLLAMA_CHAT_MODEL

# ─── JSON extraction ──────────────────────────────────────────

def _extract_json_array(raw: str) -> list:
    """Robustly extract a JSON array from messy LLM output."""
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in response: {raw[:300]!r}")
    raw = raw[start:end]
    VALID = set('"\\' + "/bfnrtu")
    def _fix(m):
        ch = m.group(1)
        return f"\\\\{ch}" if ch not in VALID else m.group(0)
    raw = re.sub(r'\\(.)', _fix, raw)
    return json.loads(raw)


# ─── content-word overlap (stopword-filtered) ────────────────

_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "of","in","to","for","on","at","by","from","with","about","as","into","through",
    "this","that","these","those","what","which","how","why","when","where","who",
    "and","or","but","if","because","while","although","however","therefore",
    "it","its","they","them","their","we","our","you","your","he","she","his","her",
    "not","no","any","all","each","both","more","most","other","some","such",
    "paper","study","work","method","approach","model","system","result","results",
    "show","shows","shown","using","used","use","based","proposed","present",
    "also","than","then","thus","here","there","between","over","under","per",
}

def _content_words(text: str) -> set[str]:
    words = re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


# ─── database loader ──────────────────────────────────────────

async def _load_benchmark_chunks() -> list[dict]:
    """Load benchmark candidates from the current PostgreSQL source of truth."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Chunk, Paper)
            .join(Paper, Chunk.paper_id == Paper.id)
            .order_by(Chunk.source, Chunk.chunk_index)
        )
        rows = result.all()

        # Map each paper to its real Abstract chunk.
        # Rows are ordered by chunk_index, so the first Abstract chunk wins
        # if a paper contains more than one.
        abstract_chunk_ids = {}
        for chunk, paper in rows:
            if (
                (chunk.section or "").strip().lower() == "abstract"
                and paper.id not in abstract_chunk_ids
            ):
                abstract_chunk_ids[paper.id] = chunk.chunk_id

        chunks = []
        for chunk, paper in rows:
            chunks.append({
                "text": chunk.text,
                "metadata": {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "paper_id": str(chunk.paper_id),
                    "title": paper.title,
                    "authors": paper.authors or chunk.authors or "",
                    "year": paper.year if paper.year is not None else chunk.year,
                    "abstract": paper.abstract or "",
                    "abstract_chunk_id": abstract_chunk_ids.get(paper.id, ""),
                    "section": chunk.section or "",
                    "section_priority": chunk.section_priority,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type or "general",
                    "content_type": chunk.content_type or "text",
                    "is_enriched": chunk.is_enriched,
                },
            })

        return chunks


# ─── QA generation ────────────────────────────────────────────

def generate_qa_from_chunk(
    chunk: dict,
    num_questions: int = 2,
) -> list[dict]:
    """Generate QA pairs anchored on the paper abstract (not the chunk text).

    Using the abstract as the generation context instead of the chunk body
    eliminates the primary source of lexical bias: the LLM can no longer
    copy rare tokens from the chunk that BM25 matches trivially.

    FIX (audit: evaluation flaw — paper-level vs chunk-level ground truth):
    Previously this function stored `source_chunk` set to the PAPER's
    filename (meta['source']), not an actual chunk_id. This meant every
    downstream Hit@k/MRR/nDCG metric only checked "did a chunk from the
    correct PAPER appear in the results" — retrieving the WRONG chunk from
    the right paper still counted as a perfect hit. The field name was
    misleading and the metric was systematically more generous than it
    should be.

    Now we store BOTH:
      - source_paper:    the paper filename (same value as before — kept
                         for backwards-compatible paper-level scoring)
      - source_chunk_id: the actual chunk_id this question was anchored to,
                         enabling a genuine chunk-level Hit@k/MRR/nDCG.
      - source_chunk:    kept as an alias of source_paper for any old code
                         that still reads this exact key name.
    """
    meta = chunk["metadata"]
    chunk_text = chunk["text"]

    if len(chunk_text.strip()) < 100:
        return []

    # Prefer abstract as the question-generation context
    abstract = meta.get("abstract", "")

    if (
        abstract
        and len(abstract.strip()) > 80
        and meta.get("abstract_chunk_id")
    ):
        generation_context = abstract[:1000]
        context_label = "Abstract"
    else:
        # Fall back to the sampled chunk whenever there is no usable abstract
        # or no real Abstract chunk to serve as exact chunk-level ground truth.
        generation_context = chunk_text[:800]
        context_label = "Section excerpt"

    prompt = f"""You are building an evaluation dataset for a RAG retrieval system.
Your goal: write questions that REQUIRE reading this paper to answer,
but that do NOT share rare vocabulary with any specific chunk of text.

Paper: {meta['title']} ({meta['year']})
{context_label}:
{generation_context}

Generate {num_questions} question-answer pairs following ALL rules below:

RULES:
1. Write questions in PLAIN ENGLISH — no symbols, no Greek letters, no LaTeX.
2. Do NOT lift phrases or technical terms from the text above into your question.
   Use synonyms, rephrase in your own words. A keyword search should NOT find
   the answer just by matching words in the question.
3. Questions must test conceptual understanding:
   "Why does X work?", "What limitation does Y address?",
   "How does Z compare to simpler approaches?", "What does W assume?"
4. Never ask for author names, dates, venues, arXiv IDs, or paper titles.
5. Answers: 1–3 sentences, grounded in the text, no LaTeX or symbols.
6. Output ONLY a valid JSON array. No prose, no markdown fences.

Output:
[
  {{"question": "...", "answer": "...", "source_chunk": "{meta['source']}", "section": "{meta['section']}"}},
  {{"question": "...", "answer": "...", "source_chunk": "{meta['source']}", "section": "{meta['section']}"}}
]"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": 0.7, "num_predict": 600},
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw   = resp.json()["message"]["content"].strip()
        pairs = _extract_json_array(raw)

        if not isinstance(pairs, list):
            return []

        chunk_cw = _content_words(chunk_text)

        BANNED = [
            "author", "arxiv", "published", "year", "venue", "conference",
            "journal", "doi", "identifier", "title of the paper", "name of the paper",
        ]
        GENERIC = [
            "primary focus", "main focus", "this section", "this paper",
            "this study", "this work", "purpose of this",
        ]

        cleaned = []
        for pair in pairs:
            q  = pair.get("question", "")
            ql = q.lower()

            if any(pat in ql for pat in BANNED):
                continue
            if any(pat in ql for pat in GENERIC):
                continue
            if re.search(r'equation\s*\(?\d+\)?|eq\.\s*\(?\d+\)?', ql):
                continue
            if re.search(r'[α-ωΑ-Ω]|\\[a-zA-Z]+\{', q):
                continue

            # Content-word overlap (stopword-filtered, stricter threshold)
            q_cw = _content_words(ql)
            if q_cw:
                overlap = len(q_cw & chunk_cw) / len(q_cw)
                if overlap > 0.35:          # tightened from 0.60
                    continue

            # Flag abstract-based questions for diagnostics
            pair["generation_source"] = context_label

            # FIX: capture real chunk-level ground truth alongside the
            # existing paper-level field. source_chunk stays as-is (paper
            # filename) for backward compatibility with any code reading
            # that exact key; source_paper is the same value under a
            # clearer name; source_chunk_id is the NEW field holding the
            # actual chunk_id this question was generated from.
            pair["source_paper"] = meta["source"]

            # Ground truth must match the text used to generate the question.
            # Abstract-generated QA points to the paper's real Abstract chunk;
            # section-generated QA points to the sampled section chunk.
            if context_label == "Abstract":
                pair["source_chunk_id"] = meta.get("abstract_chunk_id", "")
            else:
                pair["source_chunk_id"] = meta.get("chunk_id", "")

            cleaned.append(pair)

        return cleaned

    except Exception as e:
        print(f"  QA generation failed for chunk: {e}")
        return []


# ─── dataset builder ──────────────────────────────────────────

async def build_qa_dataset(
    output_path: str = "indexes/qa_dataset.json",
    target_count: int = 10,
    sample_every: int = 3,
) -> list[dict]:

    chunks = await _load_benchmark_chunks()
    print(f"Loaded {len(chunks)} chunks from current PostgreSQL DB")

    # Include abstract/intro chunks now — they're the best source for
    # paper-level questions. Exclude only reference lists.
    SKIP_SECTIONS = {"references", "bibliography", "acknowledgements"}
    valid_chunks = [
        c for c in chunks
        if c["metadata"]["section"].lower() not in SKIP_SECTIONS
        and len(c["text"].strip()) > 150
    ]

    # Shuffle for cross-paper diversity
    random.seed(42)
    random.shuffle(valid_chunks)
    sampled = valid_chunks[::sample_every]

    # Track per-paper counts to prevent one paper dominating
    paper_counts: dict[str, int] = defaultdict(int)
    MAX_PER_PAPER = 6

    print(f"Building QA dataset from {len(sampled)} sampled chunks...")
    print(f"Target: {target_count} QA pairs")
    print(f"Model:  {OLLAMA_MODEL} (local Ollama)\n")

    all_qa = []

    for i, chunk in enumerate(sampled):
        if len(all_qa) >= target_count:
            break

        paper_key = chunk["metadata"]["source"]
        if paper_counts[paper_key] >= MAX_PER_PAPER:
            continue

        print(f"[{i+1}/{len(sampled)}] {paper_key[:40]} | {chunk['metadata']['section']}")

        pairs = generate_qa_from_chunk(chunk, num_questions=2)
        if pairs:
            all_qa.extend(pairs)
            paper_counts[paper_key] += len(pairs)
        else:
            print("  (skipped — no valid pairs after filtering)")

    all_qa = all_qa[:target_count]

    # Derive summary statistics from the final saved dataset, not pre-truncation candidates.
    abstract_sourced = sum(
        q.get("generation_source") == "Abstract"
        for q in all_qa
    )
    section_sourced = len(all_qa) - abstract_sourced
    papers_covered = len({
        q.get("source_paper")
        for q in all_qa
        if q.get("source_paper")
    })

    Path("indexes").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_qa, f, indent=2)

    print(f"\nDataset saved: {len(all_qa)} QA pairs → {output_path}")
    print(f"  Abstract-anchored: {abstract_sourced} ({100*abstract_sourced//max(len(all_qa),1)}%)")
    print(f"  Section-anchored:  {section_sourced}    ({100*section_sourced//max(len(all_qa),1)}%)")
    print(f"  Papers covered:    {papers_covered}")
    return all_qa


if __name__ == "__main__":
    asyncio.run(build_qa_dataset())
