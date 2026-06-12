"""
Generate captions and descriptions for figures using Ollama VLM.
Gracefully degrades if Ollama is unavailable.
"""
from __future__ import annotations
import base64
import os
from pathlib import Path


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VLM_MODEL = "llava:7b"


def caption_figure(image_path: str, paper_title: str, figure_number: str) -> str:
    """
    Generate a detailed caption for a figure using Qwen2-VL via Ollama.
    Returns placeholder string on failure.
    """
    if not image_path or not Path(image_path).exists():
        return f"{figure_number} — image not available"

    try:
        return _caption_with_ollama(image_path, paper_title, figure_number)
    except Exception as e:
        print(f"Ollama unavailable for {figure_number}, trying Groq...")
        from src.ingestion.figure_captioner_groq import caption_figure_groq
        return caption_figure_groq(image_path, paper_title, figure_number)


def _caption_with_ollama(image_path: str, paper_title: str, figure_number: str) -> str:
    import httpx

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    prompt = f"""This is {figure_number} from the research paper "{paper_title}".

Describe this figure in detail:
1. What type of visualization is this? (architecture diagram, chart, graph, table, flowchart, etc.)
2. What are the main components or elements shown?
3. What is the key finding or information conveyed?
4. Are there any specific numbers, labels, or measurements visible?

Provide a detailed description in 3-5 sentences."""

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": VLM_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def is_ollama_available() -> bool:
    try:
        import httpx
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False
