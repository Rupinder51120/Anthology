"""
Centralized AI Model Registry

This file contains every AI model identifier used throughout the project.
Runtime code should import model names from here instead of hardcoding them.
"""
from api.core.config import get_settings

settings = get_settings()

OLLAMA_CHAT_MODEL = settings.ollama_model
GROQ_CHAT_MODEL = settings.groq_model
EMBEDDING_MODEL = settings.embedding_model
CROSS_ENCODER_MODEL = settings.cross_encoder_model
COHERE_RERANK_MODEL = settings.cohere_rerank_model
GROQ_VISION_MODEL = settings.groq_vision_model
DEPLOT_MODEL = settings.deplot_model