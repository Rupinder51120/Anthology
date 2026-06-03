import json
import random
import re
from pathlib import Path

import requests

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"


# ─── JSON extraction ──────────────────────────────────────────

def _extract_json_array(raw: str) -> list:
    """Robustly extract a JSON array from messy LLM output.

    Handles:
    - Markdown fences (```json ... ```)
    - Leading/trailing prose around the array
    - Invalid backslash escapes from LaTeX/math (\\alpha, \\frac, etc.)
      which break json.loads even though the content is fine
    """
    # Strip fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Grab everything from first [ to last ]
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in response: {raw[:300]!r}")
    raw = raw[start:end]

    # Fix lone backslashes that aren't valid JSON escape sequences.
    # JSON only allows: \" \\ \/ \b \f \n \r \t \uXXXX
    # LaTeX like \alpha, \frac, \times will cause json.loads to fail.
    VALID = set('"\\' + "/bfnrtu")
    def _fix(m):
        ch = m.group(1)
        return f"\\\\{ch}" if ch not in VALID else m.group(0)
    raw = re.sub(r'\\(.)', _fix, raw)

    return json.loads(raw)


# ─── QA generation ────────────────────────────────────────────

def generate_qa_from_chunk(chunk: dict, num_questions: int = 2) -> list[dict]:
    text = chunk["text"]
    meta = chunk["metadata"]

    if len(text.strip()) < 100:
        return []

    prompt = f"""You are building an evaluation dataset for a semantic search / RAG system.

Given this excerpt from a research paper, generate {num_questions} question-answer pairs
that test UNDERSTANDING, not lookup.

Paper: {meta['title']} ({meta['year']})
Section: {meta['section']}
Excerpt:
{text[:800]}

STRICT RULES — violating any of these makes the dataset useless:

1. NEVER ask for: author names, publication dates, arXiv IDs, paper titles,
   venue names, or any other bibliographic metadata.
2. NEVER copy phrases verbatim from the excerpt into the question.
   Rephrase using different vocabulary — a BM25 keyword search on the question
   must NOT trivially find the answer chunk.
3. Questions must require understanding the concept, not just locating a string.
   Good types: "Why does X work?", "What problem does Y solve?",
   "How does Z differ from the naive approach?", "What assumption does W make?"
4. Answers must be 1-3 sentences, factual, grounded only in the excerpt.
5. Output ONLY a valid JSON array — no explanation, no markdown fences.
   Do NOT use LaTeX or special characters in your answers — write math in plain English.

Output format:
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

        # Post-filter: drop metadata questions and copy-paste questions
        BANNED = [
            "author", "arxiv", "published", "year", "venue", "conference",
            "journal", "doi", "identifier", "title of the paper", "name of the paper",
        ]
        chunk_words = set(text.lower().split())
        cleaned = []
        for pair in pairs:
            q = pair.get("question", "").lower()
            if any(pat in q for pat in BANNED):
                continue
            q_words = set(q.split())
            if q_words and len(q_words & chunk_words) / len(q_words) > 0.6:
                continue
            cleaned.append(pair)
        return cleaned

    except Exception as e:
        print(f"  QA generation failed for chunk: {e}")
        return []


# ─── dataset builder ──────────────────────────────────────────

def build_qa_dataset(
    chunks_path:  str = "indexes/chunks_metadata.json",
    output_path:  str = "indexes/qa_dataset.json",
    target_count: int = 50,
    sample_every: int = 3,
) -> list[dict]:

    with open(chunks_path) as f:
        chunks = json.load(f)

    SKIP_SECTIONS = {"references", "abstract", "introduction", "author", "authors"}
    valid_chunks = [
        c for c in chunks
        if c["metadata"]["section"].lower() not in SKIP_SECTIONS
        and len(c["text"].strip()) > 150
    ]

    # Shuffle so we get cross-paper diversity from chunk 1, not one paper at a time
    random.seed(42)
    random.shuffle(valid_chunks)
    sampled = valid_chunks[::sample_every]

    print(f"Building QA dataset from {len(sampled)} sampled chunks...")
    print(f"Target: {target_count} QA pairs")
    print(f"Model:  {OLLAMA_MODEL} (local Ollama — no rate limits)\n")

    all_qa = []
    for i, chunk in enumerate(sampled):
        if len(all_qa) >= target_count:
            break

        print(f"[{i+1}/{len(sampled)}] {chunk['metadata']['source'][:40]} "
              f"| {chunk['metadata']['section']}")

        pairs = generate_qa_from_chunk(chunk, num_questions=2)
        if pairs:
            all_qa.extend(pairs)
        else:
            print("  (skipped — no valid pairs after filtering)")

    all_qa = all_qa[:target_count]

    Path("indexes").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_qa, f, indent=2)

    print(f"\nDataset saved: {len(all_qa)} QA pairs → {output_path}")
    return all_qa


if __name__ == "__main__":
    build_qa_dataset()