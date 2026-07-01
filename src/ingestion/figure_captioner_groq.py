
"""
Groq-based figure captioner using base64 image input.
Fallback when Ollama unavailable.
"""
import base64
import os
import time

from dotenv import load_dotenv
from pathlib import Path
from api.core.models import GROQ_VISION_MODEL
from api.core.config import get_settings

settings = get_settings()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB limit to avoid OOM and API payload errors

def caption_figure_groq(image_path: str, paper_title: str, figure_number: str) -> str:
    if not image_path or not Path(image_path).exists():
        return f"{figure_number} from {paper_title}"

    # Guard against oversized images
    try:
        if os.path.getsize(image_path) > MAX_IMAGE_SIZE:
            print(f"Figure {figure_number} too large ({os.path.getsize(image_path)} bytes). Skipping Groq.")
            return f"{figure_number} from {paper_title}"
    except Exception as e:
        print(f"Error checking file size for {image_path}: {e}")
        return f"{figure_number} from {paper_title}"

    try:
        from groq import Groq, RateLimitError, APIStatusError, APIConnectionError, AuthenticationError
        client = Groq(api_key=settings.groq_api_key.get_secret_value())

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        fallback_caption = f"{figure_number} from {paper_title}"

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=GROQ_VISION_MODEL,
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
                    timeout=30.0,
                )

                if not response.choices:
                    return fallback_caption

                content = response.choices[0].message.content
                return content.strip() if content else fallback_caption

            except AuthenticationError:
                print(f"Groq Auth failure: Check GROQ_API_KEY. Permanent error.")
                return fallback_caption
            except RateLimitError:
                print(f"Groq rate limit hit (attempt {attempt+1}/3). Backing off...")
            except (APIConnectionError, APIStatusError) as e:
                # Retry on 5xx or connection issues, but not 4xx (except 429 handled above)
                status_code = getattr(e, 'status_code', None)
                if status_code and status_code < 500 and status_code != 429:
                    print(f"Groq API permanent error {status_code}: {e}")
                    return fallback_caption
                print(f"Groq transient error (attempt {attempt+1}/3): {e}")
            except Exception as e:
                print(f"Unexpected Groq error: {e}")
                return fallback_caption

            if attempt < 2:
                time.sleep(2 ** (attempt + 1))

        return fallback_caption

    except Exception as e:
        print(f"Groq captioning critical failure: {e}")
        return f"{figure_number} from {paper_title}"
