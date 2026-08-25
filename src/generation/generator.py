"""
src/generation/generator.py

LLM providers (configurable/swappable via USE_GROQ in .env, not an
automatic runtime failover — a Groq failure surfaces as an error, it
does not retry against Ollama):
- Groq (cloud) — USE_GROQ=true
- Ollama (local) — USE_GROQ=false
"""

import json
import os
import asyncio
import logging
import requests
from typing import AsyncIterator
from dotenv import load_dotenv
from api.core.models import OLLAMA_CHAT_MODEL
from api.core.models import GROQ_CHAT_MODEL
from api.core.models import GROQ_VISION_MODEL
from api.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL      = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_MODEL    = OLLAMA_CHAT_MODEL

SYSTEM_PROMPT = """You are an expert AI research assistant helping a student understand research papers.

Answer the question directly and clearly using ONLY the provided context.

Guidelines:
- Match your answer's length and structure to the question. A simple factual
  question gets a direct 1-3 sentence answer. A question asking to explain,
  compare, or synthesize multiple sources gets a fuller explanation with
  headers only if that genuinely helps readability — do not force structure
  onto simple answers.
- Cite papers inline as you use them, e.g. "(Smith et al., 2023)" or
  "according to the Quadrangle Attention paper". Do NOT repeat a separate
  citation list after the answer — inline citation is enough.
- Never invent a numbered citation marker like "[Source 2]" or "Source 3"
  — always cite by the paper's actual title/author instead.
- Use technical terms correctly but explain them briefly if they're central
  to the answer. Don't pad with analogies unless they clarify something
  genuinely hard to grasp.
- If context is insufficient to answer, say exactly what's missing instead
  of guessing.
- Never hallucinate citations, numbers, or results not present in context.
- Use markdown only where it aids clarity (e.g. a short list for multiple
  distinct points) — not as default formatting."""


# ── Groq ──────────────────────────────────────────────────────────────────────

def _groq_enabled() -> bool:
    return settings.use_groq


def _groq_client():
    from groq import Groq
    key = settings.groq_api_key.get_secret_value()
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=key, max_retries=0)


def _call_groq(messages: list[dict]) -> str:
    client = _groq_client()
    model  = GROQ_CHAT_MODEL
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
        model=GROQ_VISION_MODEL,
        messages=vision_msgs,
        # qwen/qwen3.6-27b (the verified vision-capable model on this
        # account) is a reasoning model that emits a <think>...</think>
        # block before its actual answer; 1024 tokens was observed (live
        # test) to sometimes truncate mid-reasoning before reaching the
        # answer, so this is set high enough for the trace to complete.
        max_tokens=2048,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()
    # Strip the reasoning trace -- users should see the answer, not the
    # model's internal chain-of-thought.
    if "<think>" in answer and "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()
    return answer


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

def format_context(chunks: list[dict], max_chars: int = 4000) -> tuple[str, list[dict]]:
    parts, total = [], 0
    used = []
    for chunk in chunks:
        meta = chunk["metadata"]
        # Deliberately no "[Source N]" numbering here -- the model was
        # observed parroting that exact label as a fake inline citation
        # (e.g. "...[Source 2]") instead of citing by paper/author as
        # instructed. Giving it only the paper's real identity to cite
        # removes the numbered token it was echoing.
        part = (
            f"[{meta.get('chunk_type','general').upper()}]\n"
            f"Paper: {meta['title']} ({meta['year']})\n"
            f"Section: {meta['section']}\n"
            f"---\n{chunk['text']}"
        )
        if total + len(part) > max_chars:
            break
        parts.append(part)
        used.append(chunk)
        total += len(part)
    return "\n\n".join(parts), used


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


def collect_image_paths(chunks: list[dict]) -> list[str]:
    """
    Pulls image file paths out of retrieved figure chunks -- shared by both
    the streaming and non-streaming query paths so there is exactly one
    place that decides which figures are eligible for vision generation.
    Only figures that were actually retrieved (already filtered/reranked by
    the retrieval pipeline) are considered -- this never scans the corpus.
    """
    return [
        c["metadata"].get("image_path")
        for c in chunks
        if c["metadata"].get("content_type") == "figure"
        and c["metadata"].get("image_path")
    ]


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

def _build_messages(query: str, chunks: list[dict], chat_history: list[dict] = None) -> tuple[str, list[dict], list[dict]]:
    context, used_chunks = format_context(chunks)
    user_msg = (
        f"Context from research papers:\n\n{context}\n\n"
        f"---\nQuestion: {query}\n\n"
        f"Answer directly using the context above. Match length to the "
        f"question's complexity — don't pad."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-4:])
    messages.append({"role": "user", "content": user_msg})
    return context, messages, used_chunks


async def generate_answer(
    query:        str,
    chunks:       list[dict],
    chat_history: list[dict] = None,
    image_paths:  list[str] | None = None,
) -> dict:
    if not chunks:
        return {"answer": "No relevant information found.", "citations": [], "chunks_used": 0, "response_type": "error", "tokens_used": 0}

    context, messages, used_chunks = _build_messages(query, chunks, chat_history)
    citations         = format_citations(used_chunks)
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
        # A hard API failure (bad model name, network error, auth failure,
        # rate limit, ...) must never be silently relabeled as "the answer
        # wasn't grounded" -- that disguises a total generation outage as a
        # normal, expected "insufficient context" response and would hide a
        # broken backend from anyone reading logs/metrics. Surface it as its
        # own response_type instead.
        logger.error("Generation call failed: %s", e, exc_info=True)
        return {"answer": "The system encountered an error while generating a response. Please try again.",
                "citations": [], "chunks_used": 0, "response_type": "error", "tokens_used": 0}

    if not _is_grounded(answer, context):
        return {"answer": "Could not find a grounded answer in your papers.", "citations": [], "chunks_used": 0, "response_type": "ungrounded", "tokens_used": 0}

    return {"answer": answer, "citations": citations, "chunks_used": len(used_chunks), "response_type": response_type, "tokens_used": tokens_used}


async def stream_answer(
    query:        str,
    chunks:       list[dict],
    chat_history: list[dict] = None,
    image_paths:  list[str] | None = None,
) -> AsyncIterator[dict]:
    """
    Async generator yielding structured SSE-ready events for the
    /query/stream endpoint. This is the single streaming implementation
    (the router does not duplicate provider-selection logic) and it
    mirrors generate_answer()'s config/citation handling exactly:
    same _groq_enabled() switch, same _build_messages()/format_citations()
    helpers, so streaming and non-streaming responses stay consistent.

    image_paths (optional): file paths of retrieved figure chunks, as
    produced by collect_image_paths(). When the active provider is Groq
    (_groq_enabled() is True) and images are present, generation is routed
    through the existing _call_groq_vision() -- the same function
    generate_answer() already uses, not a second implementation. Groq's
    vision response isn't token-streamed here (the underlying call is a
    single blocking request); it is yielded as one "token" event once
    ready, so the SSE contract (status -> token(s) -> citations -> [DONE])
    is unchanged for the client. When the active provider is Ollama, the
    configured local model (qwen2.5:7b) is not vision-capable, so images
    are intentionally ignored and generation falls back to the normal
    text-only streaming path below -- silently pretending a non-vision
    model can see images would be worse than just not using them.

    Yields dicts of the form:
      {"type": "status",    "text": str}                -- optional, informational
      {"type": "token",     "text": str}
      {"type": "citations", "citations": list[dict]}   -- always last on success
      {"type": "error",     "text": str}                -- terminal on failure
    """
    if not chunks:
        yield {"type": "token", "text": "No relevant information found."}
        yield {"type": "citations", "citations": []}
        return

    context, messages, used_chunks = _build_messages(query, chunks, chat_history)
    citations = format_citations(used_chunks)

    if _groq_enabled() and image_paths:
        try:
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, _call_groq_vision, messages, image_paths)
            yield {"type": "token", "text": answer}
            yield {"type": "citations", "citations": citations}
            return
        except Exception as e:
            # The vision call can fail for reasons a missing/invalid image
            # file doesn't cover -- e.g. the configured vision model isn't
            # actually available on the current provider account/key. That
            # is a real, observed failure mode (see README), not a
            # hypothetical: don't let it silently pretend to work, but
            # also don't kill the whole answer -- fall back to the normal
            # text-only path below instead of erroring the request.
            logger.error("Vision generation call failed, falling back to text-only: %s", e, exc_info=True)
            yield {"type": "status", "text": "Vision generation was unavailable for the retrieved figure(s) -- answering from text only."}

    try:
        if _groq_enabled():
            from groq import AsyncGroq
            client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())
            stream = await client.chat.completions.create(
                model=GROQ_CHAT_MODEL, messages=messages, max_tokens=1024, temperature=0.2, stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield {"type": "token", "text": delta}
            await client.close()
        else:
            if image_paths:
                yield {"type": "status", "text": "Relevant figures were found, but the local generation model does not support vision -- answering from text only."}
            import httpx
            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream(
                    "POST", OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL, "messages": messages, "stream": True,
                        "options": {"temperature": 0.2, "num_predict": 1024, "num_ctx": 8192},
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data  = json.loads(line)
                        delta = data.get("message", {}).get("content", "")
                        if delta:
                            yield {"type": "token", "text": delta}
                        if data.get("done"):
                            break
    except Exception as e:
        logger.error("Streaming generation call failed: %s", e, exc_info=True)
        yield {"type": "error", "text": "The system encountered an error while generating a response."}
        return

    yield {"type": "citations", "citations": citations}
