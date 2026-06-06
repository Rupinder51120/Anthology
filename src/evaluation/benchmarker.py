import json
import random
import re
from pathlib import Path
from collections import defaultdict

import requests

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

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


# ─── registry loader ──────────────────────────────────────────

def _load_registry(registry_path: str = "data/download_registry.json") -> dict[str, dict]:
    """Return arxiv_id → {title, abstract, year, authors} from the registry."""
    try:
        with open(registry_path) as f:
            registry = json.load(f)
        # Registry is a dict keyed by arxiv_id
        if isinstance(registry, dict):
            return registry
        # Fallback: list of records
        if isinstance(registry, list):
            return {r["arxiv_id"]: r for r in registry if "arxiv_id" in r}
    except FileNotFoundError:
        pass
    return {}


def _abstract_for_chunk(chunk: dict, registry: dict) -> str | None:
    """Return the abstract for the paper this chunk came from, or None."""
    source = chunk["metadata"].get("source", "")
    # Registry is keyed by arxiv_id, but each entry has a filename field.
    # Build a filename→entry lookup on first call.
    if not hasattr(_abstract_for_chunk, "_filename_index"):
        _abstract_for_chunk._filename_index = {
            v["filename"]: v for v in registry.values() if "filename" in v
        }
    entry = _abstract_for_chunk._filename_index.get(source)
    return entry["abstract"].strip() if entry and entry.get("abstract") else None


# ─── QA generation ────────────────────────────────────────────

def generate_qa_from_chunk(
    chunk: dict,
    num_questions: int = 2,
    registry: dict | None = None,
) -> list[dict]:
    """Generate QA pairs anchored on the paper abstract (not the chunk text).

    Using the abstract as the generation context instead of the chunk body
    eliminates the primary source of lexical bias: the LLM can no longer
    copy rare tokens from the chunk that BM25 matches trivially.

    The chunk is still recorded as source_chunk so the evaluator knows
    which chunk to look for at retrieval time.
    """
    meta = chunk["metadata"]
    chunk_text = chunk["text"]

    if len(chunk_text.strip()) < 100:
        return []

    # Prefer abstract as the question-generation context
    abstract = None
    if registry is not None:
        abstract = _abstract_for_chunk(chunk, registry)

    if abstract and len(abstract.strip()) > 80:
        generation_context = abstract[:1000]
        context_label = "Abstract"
    else:
        # Fallback: use the chunk, but the prompt now explicitly forces paraphrase
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
            cleaned.append(pair)

        return cleaned

    except Exception as e:
        print(f"  QA generation failed for chunk: {e}")
        return []


# ─── dataset builder ──────────────────────────────────────────

def build_qa_dataset(
    chunks_path:   str = "indexes/chunks_metadata.json",
    registry_path: str = "data/download_registry.json",
    output_path:   str = "indexes/qa_dataset.json",
    target_count:  int = 100,       # raised: 50 is below statistical validity threshold
    sample_every:  int = 3,
) -> list[dict]:

    with open(chunks_path) as f:
        chunks = json.load(f)

    registry = _load_registry(registry_path)
    print(f"Registry loaded: {len(registry)} papers")

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
    abstract_sourced = 0
    chunk_sourced = 0

    for i, chunk in enumerate(sampled):
        if len(all_qa) >= target_count:
            break

        paper_key = chunk["metadata"]["source"]
        if paper_counts[paper_key] >= MAX_PER_PAPER:
            continue

        print(f"[{i+1}/{len(sampled)}] {paper_key[:40]} | {chunk['metadata']['section']}")

        pairs = generate_qa_from_chunk(chunk, num_questions=2, registry=registry)
        if pairs:
            for p in pairs:
                if p.get("generation_source") == "Abstract":
                    abstract_sourced += 1
                else:
                    chunk_sourced += 1
            all_qa.extend(pairs)
            paper_counts[paper_key] += len(pairs)
        else:
            print("  (skipped — no valid pairs after filtering)")

    all_qa = all_qa[:target_count]

    Path("indexes").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_qa, f, indent=2)

    print(f"\nDataset saved: {len(all_qa)} QA pairs → {output_path}")
    print(f"  Abstract-anchored: {abstract_sourced} ({100*abstract_sourced//max(len(all_qa),1)}%)")
    print(f"  Chunk-anchored:    {chunk_sourced}    ({100*chunk_sourced//max(len(all_qa),1)}%)")
    print(f"  Papers covered:    {len(paper_counts)}")
    return all_qa


if __name__ == "__main__":
    build_qa_dataset()