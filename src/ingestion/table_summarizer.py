"""
Generate natural language summaries for extracted tables using Ollama qwen2.5:7b.
"""
from __future__ import annotations
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def summarize_table(table_markdown: str, paper_title: str) -> str:
    try:
        import httpx
        prompt = f"""You are analyzing a table from the research paper "{paper_title}".
Table:
{table_markdown}
Write a 2-3 sentence factual summary of what this table shows.
Focus on key numbers, comparisons, and findings.
Be specific. Do not say "the table shows" — just state the findings directly."""
        response = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"Table summarization failed: {e}")
        return ""
