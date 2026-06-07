from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from api.schemas.schemas import TTSRequest

router = APIRouter(prefix="/api/v1", tags=["TTS"])


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech — returns WAV audio bytes."""
    from src.ui.tts import text_to_speech as tts

    audio = tts(request.text, voice=request.voice, rate=request.rate)
    if not audio:
        raise HTTPException(
            status_code=500,
            detail="TTS failed — macOS say command not available"
        )

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=answer.wav"},
    )
