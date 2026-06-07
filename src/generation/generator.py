"""
src/generation/generator.py

Migrated from Groq to Ollama (local LLM).
- No API key needed, no rate limits
- Model: qwen2.5:7b (runs on 16GB Mac)
- Streaming works via Ollama stream API
"""

import json
import requests

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """You are an expert AI research assistant helping a student deeply understand research papers.

Given context excerpts from research papers, write a COMPLETE and THOROUGH answer.

Structure your response as:

**EXPLANATION**
Explain the concept fully. Cover all key points from the context — do not stop early.
Break down technical terms. Use analogies where helpful.
If multiple papers address the question, synthesize all of them.

**KEY INSIGHT**
One sentence capturing the single most important idea.

**EVIDENCE FROM PAPERS**
For EACH source used, write 2-3 sentences summarizing what that specific paper contributes.
Format: "Paper Title (Year): ..."
Cover every source that is relevant — do not skip any.

Rules:
- ONLY use information from the provided context
- A complete answer covers ALL relevant points in the context, not just the first one
- If context is insufficient, state exactly what is missing
- Never hallucinate citations or results
- Use markdown formatting

MANDATORY FINAL SECTION — always end with:
## Sources Used
For every source in the context write exactly one bullet:
- [Paper Title (Year)]: [one sentence on what it contributes to this answer]
Cover EVERY source. Do not skip any."""


def format_context(chunks: list[dict]) -> str:
    """Format chunks into a numbered context block for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        chunk_type = meta.get("chunk_type", "general").upper()
        parts.append(
            f"[Source {i}] [{chunk_type}]\n"
            f"Paper: {meta['title']} ({meta['year']})\n"
            f"Section: {meta['section']}\n"
            f"---\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def format_citations(chunks: list[dict]) -> list[dict]:
    seen      = set()
    citations = []
    for chunk in chunks:
        meta = chunk["metadata"]
        key  = (meta["title"], meta["section"])
        if key not in seen:
            seen.add(key)
            citations.append({
                "title":    meta["title"],
                "authors":  meta["authors"],
                "year":     meta["year"],
                "section":  meta["section"],
                "filename": meta["source"],
                "doi":      meta.get("doi"),
                "score":    meta.get("rerank_score"),
            })
    citations.sort(key=lambda x: x.get("score") or 0, reverse=True)
    return citations


def detect_response_type(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["recommend", "find", "suggest", "which paper"]):
        return "recommendation"
    if any(w in q for w in ["compare", "difference", "versus", "vs"]):
        return "comparison"
    return "explanation"


def _call_ollama(messages: list[dict], stream: bool = False) -> requests.Response:
    return requests.post(
        OLLAMA_URL,
        json={
            "model":   OLLAMA_MODEL,
            "messages": messages,
            "stream":  stream,
            "options": {
                "temperature": 0.2,
                "num_predict": 3000,
                "num_ctx":     16384,  # raised — prevents context truncation
            },
        },
        stream=stream,
        timeout=180,       # raised from 120 — longer answers need more time
    )


def generate_answer(
    query:        str,
    chunks:       list[dict],
    chat_history: list[dict] = None,
) -> dict:
    if not chunks:
        return {
            "answer":        "I couldn't find relevant information in your papers for this query.",
            "citations":     [],
            "chunks_used":   0,
            "response_type": "error",
            "tokens_used":   0,
        }

    context       = format_context(chunks)
    citations     = format_citations(chunks)
    response_type = detect_response_type(query)

    user_message = (
        f"Context from research papers:\n\n{context}\n\n"
        f"---\n"
        f"Question: {query}\n\n"
        f"Write a thorough, complete answer using ALL relevant sources above. "
        f"Do not stop after covering just one source — synthesize everything relevant. "
        f"Cite each paper by title and year."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-4:])   # reduced from 6 to free context space
    messages.append({"role": "user", "content": user_message})

    try:
        resp = _call_ollama(messages, stream=False)
        resp.raise_for_status()
        data        = resp.json()
        answer      = data["message"]["content"].strip()
        tokens_used = data.get("eval_count", 0)
    except Exception as e:
        answer      = f"Generation failed: {e}"
        tokens_used = 0

    return {
        "answer":        answer,
        "citations":     citations,
        "chunks_used":   len(chunks),
        "response_type": response_type,
        "tokens_used":   tokens_used,
    }


def generate_answer_streaming(
    query:        str,
    chunks:       list[dict],
    chat_history: list[dict] = None,
):
    """Yields answer tokens one by one. Use with st.write_stream() in Streamlit."""
    if not chunks:
        yield "I couldn't find relevant information in your papers for this query."
        return

    context = format_context(chunks)
    user_message = (
        f"Context from research papers:\n\n{context}\n\n"
        f"---\n"
        f"Question: {query}\n\n"
        f"Write a thorough, complete answer using ALL relevant sources above. "
        f"Do not stop after covering just one source — synthesize everything relevant. "
        f"Cite each paper by title and year."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-4:])
    messages.append({"role": "user", "content": user_message})

    try:
        resp = _call_ollama(messages, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data  = json.loads(line)
                delta = data.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if data.get("done"):
                    break
    except Exception as e:
        yield f"Generation failed: {e}"