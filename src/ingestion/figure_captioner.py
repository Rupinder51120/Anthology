"""
Generate captions and descriptions for figures using Ollama VLM.
Gracefully degrades if Ollama is unavailable.
"""
from __future__ import annotations
import os
from pathlib import Path


def caption_figure(image_path: str, paper_title: str, figure_number: str) -> dict:
    """
    Try DePlot for charts, fall back to Groq vision for other figures.
    Returns dict with keys: caption, table_data (if chart)
    """
    if not image_path or not Path(image_path).exists():
        return {"caption": f"{figure_number} — image not available", "table_data": None}
    try:
        from src.ingestion.graph_parser import parse_chart
        table_data = parse_chart(image_path)
        if table_data:
            return {"caption": f"{figure_number}: Chart data extracted.", "table_data": table_data}
    except Exception as e:
        print(f"DePlot fallback failure for {image_path}: {e}")
    from src.ingestion.figure_captioner_groq import caption_figure_groq
    caption = caption_figure_groq(image_path, paper_title, figure_number)
    return {"caption": caption, "table_data": None}
