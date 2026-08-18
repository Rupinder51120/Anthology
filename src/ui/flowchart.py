import os
import requests
import re
from api.core.models import OLLAMA_CHAT_MODEL

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL      = f"{OLLAMA_BASE_URL}/api/generate"

FLOWCHART_PROMPT = """You are an expert technical diagram engineer. Convert this explanation into a clean, valid Mermaid.js diagram.

Question: {query}
Explanation: {explanation}

STRICT RULES:
1. Output ONLY the mermaid code block. No intro text, no explanation outside the block.
2. Start with ```mermaid and end with ```
3. Use "graph TD" for processes/flows, "graph LR" for pipelines, "sequenceDiagram" for step-by-step interactions
4. Maximum 12 nodes. Keep labels SHORT (2-5 words max per node).
5. Quote ALL labels that contain special characters, parentheses, brackets, math, or punctuation. Example: A["Loss Function (MSE)"] --> B["Optimizer"]
6. No HTML tags inside labels. No raw symbols like <, >, &.
7. Every node must be reachable — no orphan nodes.
8. Use subgraphs only if there are clearly 2+ distinct phases.
9. Add brief edge labels on key transitions using |label| syntax.

Output ONLY the mermaid block:"""


def _sanitize_mermaid(raw: str) -> str:
    """Fix common LLM Mermaid mistakes."""
    lines = raw.split('\n')
    clean = []
    for line in lines:
        # Remove HTML tags from labels
        line = re.sub(r'<[^>]+>', '', line)
        # Fix unquoted labels with parens: A(text with (parens)) -> A["text with parens"]
        line = re.sub(r'\[([^\]"]*[\(\)][^\]"]*)\]', lambda m: f'["{m.group(1)}"]', line)
        clean.append(line)
    return '\n'.join(clean)


def generate_flowchart(query: str, explanation: str) -> str | None:
    if not query or not explanation:
        return None

    # Trim explanation to most relevant part
    explanation_trimmed = explanation[:2000]

    prompt = FLOWCHART_PROMPT.format(
        query=query,
        explanation=explanation_trimmed
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.05,  # near-deterministic for valid syntax
                    "num_predict": 600
                }
            },
            timeout=90
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Primary: extract fenced mermaid block
        match = re.search(r'```mermaid\s*\n([\s\S]*?)\n\s*```', raw)
        if match:
            diagram = _sanitize_mermaid(match.group(1).strip())
            return f"```mermaid\n{diagram}\n```"

        # Fallback: look for graph TD / graph LR / sequenceDiagram
        for keyword in ["graph TD", "graph LR", "graph RL", "sequenceDiagram", "flowchart TD"]:
            if keyword in raw:
                start = raw.find(keyword)
                end_match = re.search(r'```', raw[start:])
                end = start + end_match.start() if end_match else len(raw)
                diagram = _sanitize_mermaid(raw[start:end].strip())
                return f"```mermaid\n{diagram}\n```"

        print(f"Flowchart: no valid diagram found in response: {raw[:200]}")
        return None

    except Exception as e:
        print(f"Flowchart exception: {e}")
        return None