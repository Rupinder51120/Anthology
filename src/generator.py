"""
src/generator.py

Migrated from Groq to Ollama (local LLM).
- No API key needed
- No rate limits
- No token limits
- Model: qwen2.5:7b (runs on 16GB Mac)
- Streaming still works via Ollama stream API
"""

import json
import requests

OLLAMA_URL  = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """You are an expert AI research assistant helping a student deeply understand research papers.

Your response must follow this structure:

**EXPLANATION**
Explain the concept clearly, step by step. Break down every technical term.
Use analogies when helpful. Teach, don't just recite.

**KEY INSIGHT**
One sentence capturing the core idea the student must remember.

**FROM THE PAPERS**
Quote or closely paraphrase the most relevant passage from the context.
Always mention which paper it came from.

Rules:
- ONLY use information from the provided context
- If context is insufficient, say exactly what's missing
- Never hallucinate citations or results
- Use markdown formatting"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        score = meta.get("rerank_score", "")
        score_str = f" [score: {score}]" if score else ""
        parts.append(
            f"[Source {i}]{score_str}\n"
            f"Paper: {meta['title']} ({meta['year']})\n"
            f"Authors: {meta['authors']}\n"
            f"Section: {meta['section']}\n"
            f"Type: {meta.get('chunk_type', 'general')}\n"
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
                "score":    meta.get("rerank_score")
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
            "model":    OLLAMA_MODEL,
            "messages": messages,
            "stream":   stream,
            "options":  {"temperature": 0.2, "num_predict": 1500},
        },
        stream=stream,
        timeout=120,
    )


def generate_answer(
    query: str,
    chunks: list[dict],
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
        f"Context from research papers:\n{context}\n\n"
        f"---\nStudent question: {query}\n\n"
        f"Answer the question thoroughly using the context above. "
        f"Always mention which paper (title + year) each piece of information comes from."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_message})

    try:
        resp   = _call_ollama(messages, stream=False)
        resp.raise_for_status()
        data   = resp.json()
        answer = data["message"]["content"].strip()
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
    query: str,
    chunks: list[dict],
    chat_history: list[dict] = None,
):
    """Yields answer tokens one by one. Use with st.write_stream() in Streamlit."""
    if not chunks:
        yield "I couldn't find relevant information in your papers for this query."
        return

    context = format_context(chunks)
    user_message = (
        f"Context from research papers:\n{context}\n\n"
        f"---\nStudent question: {query}\n\n"
        f"Answer the question thoroughly using the context above. "
        f"Always mention which paper (title + year) each piece of information comes from."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": user_message})

    try:
        resp = _call_ollama(messages, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if chunk.get("done"):
                    break
    except Exception as e:
        yield f"Generation failed: {e}"