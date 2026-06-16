from fastapi import APIRouter
from pydantic import BaseModel
import os, json

router = APIRouter(prefix="/api/v1", tags=["Suggest"])

class SuggestRequest(BaseModel):
    question: str
    answer: str

@router.post("/suggest")
async def suggest(req: SuggestRequest):
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", ""))
        model  = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
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
