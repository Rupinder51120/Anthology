import os
import json
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_qa_from_chunk(chunk: dict, num_questions: int = 2) -> list[dict]:
    text = chunk["text"]
    meta = chunk["metadata"]

    if len(text.strip()) < 100:
        return []

    prompt = f"""You are creating an evaluation dataset for a RAG system.

Given this excerpt from a research paper, generate {num_questions} question-answer pairs.

Paper: {meta['title']} ({meta['year']})
Section: {meta['section']}
Excerpt:
{text[:800]}

Rules:
- Questions must be answerable ONLY from the excerpt above
- Answers must be factual, specific, 1-3 sentences
- Questions should test understanding, not just recall
- Output ONLY valid JSON, no explanation

Output format:
[
  {{"question": "...", "answer": "...", "source_chunk": "{meta['source']}", "section": "{meta['section']}"}},
  {{"question": "...", "answer": "...", "source_chunk": "{meta['source']}", "section": "{meta['section']}"}}
]"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        raw = response.choices[0].message.content.strip()

        # clean json fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        pairs = json.loads(raw)
        return pairs if isinstance(pairs, list) else []

    except Exception as e:
        print(f"  QA generation failed for chunk: {e}")
        return []


def build_qa_dataset(
    chunks_path: str = "indexes/chunks_metadata.json",
    output_path: str = "indexes/qa_dataset.json",
    target_count: int = 50,
    sample_every: int = 3
) -> list[dict]:

    with open(chunks_path) as f:
        chunks = json.load(f)

    # skip references and very short chunks
    valid_chunks = [
        c for c in chunks
        if c["metadata"]["section"] != "references"
        and len(c["text"].strip()) > 150
    ]

    # sample every Nth chunk for diversity
    sampled = valid_chunks[::sample_every]

    print(f"Building QA dataset from {len(sampled)} sampled chunks...")
    print(f"Target: {target_count} QA pairs\n")

    all_qa = []
    for i, chunk in enumerate(sampled):
        if len(all_qa) >= target_count:
            break

        print(f"[{i+1}/{len(sampled)}] Generating from: {chunk['metadata']['source'][:40]} | {chunk['metadata']['section']}")
        pairs = generate_qa_from_chunk(chunk, num_questions=2)
        all_qa.extend(pairs)

        # rate limit: groq free tier
        time.sleep(0.5)

    all_qa = all_qa[:target_count]

    Path("indexes").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_qa, f, indent=2)

    print(f"\nDataset saved: {len(all_qa)} QA pairs → {output_path}")
    return all_qa


if __name__ == "__main__":
    build_qa_dataset()