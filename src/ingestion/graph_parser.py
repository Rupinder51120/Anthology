"""
Graph/chart parsing using DePlot (google/deplot).
Extracts structured data tables from scientific charts.
Falls back to Groq vision for non-chart figures.
"""
from __future__ import annotations
import os
from pathlib import Path

_deplot_processor = None
_deplot_model = None

def _get_deplot():
    global _deplot_processor, _deplot_model
    if _deplot_processor is None:
        from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
        _deplot_processor = Pix2StructProcessor.from_pretrained('google/deplot')
        _deplot_model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    return _deplot_processor, _deplot_model

def parse_chart(image_path: str) -> str | None:
    """
    Extract data table from a chart image using DePlot.
    Returns markdown table string or None if not a chart.
    """
    if not image_path or not Path(image_path).exists():
        return None
    try:
        from PIL import Image

        # 1. Model Load
        try:
            processor, model = _get_deplot()
        except Exception as e:
            print(f"DePlot model load failed: {e}")
            return None

        # 2. Image Load and Validation
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB') # Load into memory for inference
        except Exception as e:
            print(f"DePlot image load failure for {image_path}: {e}")
            return None

        # 3. Inference
        try:
            inputs = processor(
                images=image,
                text='Generate underlying data table of the figure below:',
                return_tensors='pt'
            )
            predictions = model.generate(**inputs, max_new_tokens=512)
            result = processor.decode(predictions[0], skip_special_tokens=True)
            result = result.replace('<0x0A>', '\n').strip()

            # If output is too short or has no numbers, likely not a chart
            if len(result) < 20 or not any(c.isdigit() for c in result):
                return None
            return result
        except Exception as e:
            print(f"DePlot inference failure for {image_path}: {e}")
            return None

    except Exception as e:
        print(f"Unexpected DePlot error for {image_path}: {e}")
        return None
