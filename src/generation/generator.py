"""
src/generation/generator.py

LLM providers:
- Groq (cloud) — USE_GROQ=true in .env
- Ollama (local) — fallback
"""

import json
import os
import asyncio
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """You are an expert AI research assistant helping a student deeply understand research papers.

Given context excerpts from research papers, write a COMPLETE and THOROUGH answer.

Structure your response as:


## EXPLANATION

Explain the concept fully. Cover all key points from the context — do not stop early.
Break down technical terms. Use analogies where helpful.
If multiple papers address the question, synthesize all of them.


## KEY INSIGHT

One sentence capturing the single most important idea.


## EVIDENCE FROM PAPERS

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


# ── Groq ──────────────────────────────────────────────────────────────────────

def _groq_enabled() -> bool:
    return os.getenv("USE_GROQ", "false").lower() == "true"


def _groq_client():
    from groq import Groq
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=key, max_retries=0)


def _call_groq(messages: list[dict]) -> str:
    client = _groq_client()
    model  = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    resp   = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _call_groq_vision(messages: list[dict], image_paths: list[str]) -> str:
    import base64
    client   = _groq_client()
    contents = []
    for p in image_paths[:3]:
        try:
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        except Exception:
            continue
    if not contents:
        return _call_groq(messages)
    last = messages[-1]
    vision_msgs = messages[:-1] + [{"role": "user", "content": contents + [{"type": "text", "text": last["content"]}]}]
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=vision_msgs,
        max_tokens=1024,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ── Ollama ────────────────────────────────────────────────────────────────────

def _call_ollama(messages: list[dict], stream: bool = False) -> requests.Response:
    return requests.post(
        OLLAMA_URL,
        json={
            "model":   OLLAMA_MODEL,
            "messages": messages,
            "stream":   stream,
            "options":  {"temperature": 0.2, "num_predict": 1024, "num_ctx": 8192},
        },
        stream=stream,
        timeout=180,
    )


# ── Context helpers ───────────────────────────────────────────────────────────

def format_context(chunks: list[dict], max_chars: int = 4000) -> str:
    parts, total = [], 0
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        part = (
            f"[Source {i}] [{meta.get('chunk_type','general').upper()}]\n"
            f"Paper: {meta['title']} ({meta['year']})\n"
            f"Section: {meta['section']}\n"
            f"---\n{chunk['text']}"
        )
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def format_citations(chunks: list[dict]) -> list[dict]:
    seen, citations = set(), []
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
    return sorted(citations, key=lambda x: x.get("score") or 0, reverse=True)


def detect_response_type(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["recommend", "find", "suggest", "which paper"]):
        return "recommendation"
    if any(w in q for w in ["compare", "difference", "versus", "vs"]):
        return "comparison"
    return "explanation"


def _is_grounded(answer: str, context: str) -> bool:
    if not answer or not context:
        return True
    ctx_words = set(w.lower() for w in context.split() if len(w) > 4)
    ans_words = set(w.lower() for w in answer.split() if len(w) > 4)
    if not ans_words:
        return True
    return len(ctx_words & ans_words) / len(ans_words) > 0.15


# ── Main generation ───────────────────────────────────────────────────────────

def _build_messages(query: str, chunks: list[dict], chat_history: list[dict] = None) -> tuple[str, list[dict]]:
    context = format_context(chunks)
    user_msg = (
        f"Context from research papers:\n\n{context}\n\n"
        f"---\nQuestion: {query}\n\n"
        f"Write a thorough, complete answer using ALL relevant sources. "
        f"Synthesize everything relevant. Cite each paper by title and year."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-4:])
    messages.append({"role": "user", "content": user_msg})
    return context, messages


async def generate_answer(
    query:        str,
    chunks:       list[dict],
    chat_history: list[dict] = None,
    image_paths:  list[str] | None = None,
) -> dict:
    if not chunks:
        return {"answer": "No relevant information found.", "citations": [], "chunks_used": 0, "response_type": "error", "tokens_used": 0}

    context, messages = _build_messages(query, chunks, chat_history)
    citations         = format_citations(chunks)
    response_type     = detect_response_type(query)

    try:
        if _groq_enabled():
            loop   = asyncio.get_running_loop()
            if image_paths:
                answer = await loop.run_in_executor(None, _call_groq_vision, messages, image_paths)
            else:
                answer = await loop.run_in_executor(None, _call_groq, messages)
            tokens_used = 0
        else:
            resp        = await asyncio.get_running_loop().run_in_executor(None, _call_ollama, messages, False)
            resp.raise_for_status()
            data        = resp.json()
            answer      = data["message"]["content"].strip()
            tokens_used = data.get("eval_count", 0)
    except Exception as e:
        answer      = f"Generation failed: {e}"
        tokens_used = 0

    if not _is_grounded(answer, context):
        return {"answer": "Could not find a grounded answer in your papers.", "citations": [], "chunks_used": 0, "response_type": "ungrounded", "tokens_used": 0}

    return {"answer": answer, "citations": citations, "chunks_used": len(chunks), "response_type": response_type, "tokens_used": tokens_used}


def generate_answer_streaming(
    query:        str,
    chunks:       list[dict],
    chat_history: list[dict] = None,
    image_paths:  list[str] | None = None,
):
    if not chunks:
        yield "No relevant information found."
        return

    _, messages = _build_messages(query, chunks, chat_history)

    try:
        if _groq_enabled():
            client     = _groq_client()
            groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            stream     = client.chat.completions.create(
                model=groq_model, messages=messages, max_tokens=1024, temperature=0.2, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
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
