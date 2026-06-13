"""
Groq-based figure captioner using base64 image input.
Fallback when Ollama unavailable.
"""
import base64
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def caption_figure_groq(image_path: str, paper_title: str, figure_number: str) -> str:
    if not image_path or not Path(image_path).exists():
        return f"{figure_number} from {paper_title}"
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"This is {figure_number} from the research paper '{paper_title}'. Describe what this figure shows in 2-3 sentences. Focus on the type of visualization, key components, and main finding."
                    }
                ],
            }],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq captioning failed: {e}")
        return f"{figure_number} from {paper_title}"
