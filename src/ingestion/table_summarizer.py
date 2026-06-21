"""
Generate natural language summaries for extracted tables using Ollama qwen2.5:7b.
"""
from __future__ import annotations
import os
import time
import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MAX_TABLE_CHARS = 10000 # Prevent payload-too-large and prompt overflow



def summarize_table(table_markdown: str, paper_title: str) -> str:
    if not table_markdown or not table_markdown.strip():
        return ""

    # Guard against oversized tables
    if len(table_markdown) > MAX_TABLE_CHARS:
        cutoff = table_markdown.rfind("\n", 0, MAX_TABLE_CHARS)
        if cutoff != -1:
            table_markdown = table_markdown[:cutoff].strip()
        else:
            table_markdown = table_markdown[:MAX_TABLE_CHARS].strip()
        table_markdown += "\n... [Table truncated]"

    prompt = f"""You are analyzing a table from the research paper "{paper_title}".
Table:
{table_markdown}
Write a 2-3 sentence factual summary of what this table shows.
Focus on key numbers, comparisons, and findings.
Be specific. Do not say "the table shows" — just state the findings directly."""

    for attempt in range(3):
        try:
            response = httpx.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False},
                timeout=60.0,
            )

            if response.status_code == 404:
                print(f"Table summarization failed: Model 'qwen2.5:7b' not found on Ollama.")
                return "" # Model missing is a permanent failure

            response.raise_for_status()
            summary = response.json().get("response", "").strip()
            if not summary:
                return ""
            return summary

        except httpx.TimeoutException:
            print(f"Table summarization timeout (attempt {attempt+1}/3)")
        except httpx.ConnectError:
            print(f"Table summarization connection failure (attempt {attempt+1}/3)")
        except httpx.HTTPStatusError as e:
            print(f"Table summarization API error {e.response.status_code} (attempt {attempt+1}/3)")
        except Exception as e:
            print(f"Table summarization unexpected error: {e}")
            return ""

        if attempt < 2:
            time.sleep(2 ** (attempt + 1))

    return ""
