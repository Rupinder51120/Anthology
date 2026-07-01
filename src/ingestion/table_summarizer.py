
from __future__ import annotations

import os
import time
import logging
from api.core.models import GROQ_CHAT_MODEL
from api.core.config import get_settings
logger = logging.getLogger(__name__)

MAX_TABLE_CHARS = 8000   # keep well under Groq's context limit
settings = get_settings()

def summarize_table(table_markdown: str, paper_title: str) -> str:
    """
    Generate a 2-3 sentence factual summary of a markdown table using Groq.
    Returns "" on failure — never raises.
    """
    if not table_markdown or not table_markdown.strip():
        return ""

    # Guard against oversized payloads
    if len(table_markdown) > MAX_TABLE_CHARS:
        cutoff = table_markdown.rfind("\n", 0, MAX_TABLE_CHARS)
        table_markdown = (
            table_markdown[:cutoff].strip()
            if cutoff != -1
            else table_markdown[:MAX_TABLE_CHARS].strip()
        )
        table_markdown += "\n... [Table truncated]"

    prompt = (
        f'You are analyzing a table from the research paper "{paper_title}".\n\n'
        f"Table:\n{table_markdown}\n\n"
        "Write a 2-3 sentence factual summary of what this table shows. "
        "Focus on key numbers, comparisons, and findings. "
        "Be specific. Do not say 'the table shows' — just state the findings directly."
    )

    api_key = settings.groq_api_key.get_secret_value()
    if not api_key:
        logger.warning("GROQ_API_KEY not set — skipping table summarisation")
        return ""

    try:
        from groq import Groq, RateLimitError, APIStatusError, APIConnectionError, AuthenticationError
        client = Groq(api_key=api_key)
    except Exception as e:
        logger.warning("Groq client init failed: %s", e)
        return ""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                timeout=30.0,
            )
            if not response.choices:
                return ""
            content = response.choices[0].message.content
            return content.strip() if content else ""

        except AuthenticationError:
            logger.error("Groq authentication failed — check GROQ_API_KEY")
            return ""
        except RateLimitError:
            logger.warning("Groq rate limit (attempt %d/3)", attempt + 1)
        except (APIConnectionError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status and status < 500 and status != 429:
                logger.warning("Groq permanent error %s: %s", status, e)
                return ""
            logger.warning("Groq transient error (attempt %d/3): %s", attempt + 1, e)
        except Exception as e:
            logger.warning("Groq unexpected error: %s", e)
            return ""

        if attempt < 2:
            time.sleep(2 ** (attempt + 1))

    return ""