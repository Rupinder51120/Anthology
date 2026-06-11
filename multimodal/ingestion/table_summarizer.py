"""
Generate natural language summaries for extracted tables using Groq.
"""
from __future__ import annotations
import os


def summarize_table(table_markdown: str, paper_title: str) -> str:
    """
    Call Groq to generate a 2-3 sentence summary of a table.
    Returns empty string on failure.
    """
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = f"""You are analyzing a table from the research paper "{paper_title}".

Table:
{table_markdown}

Write a 2-3 sentence factual summary of what this table shows. 
Focus on key numbers, comparisons, and findings.
Be specific. Do not say "the table shows" — just state the findings directly."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Table summarization failed: {e}")
        return ""
