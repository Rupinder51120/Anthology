from fastapi import APIRouter
from pydantic import BaseModel
import os, json
from api.core.models import GROQ_CHAT_MODEL
from api.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/api/v1", tags=["Suggest"])

class SuggestRequest(BaseModel):
    question: str
    answer: str

@router.post("/suggest")
async def suggest(req: SuggestRequest):
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())
        model = GROQ_CHAT_MODEL
        resp = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    f"Given this research Q&A, return ONLY a JSON array of 3 short follow-up questions. "
                    f"No explanation, no markdown, just the array.\n\nQ: {req.question}\nA: {req.answer}"
                ),
            }],
            max_tokens=200,
            temperature=0.7,
        )
        await client.close()
        text = resp.choices[0].message.content.strip()
        arr  = json.loads(text.replace("```json","").replace("```","").strip())
        return {"suggestions": arr[:3] if isinstance(arr, list) else []}
    except Exception:
        return {"suggestions": []}
