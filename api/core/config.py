from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Anthology API"
    app_version: str = "1.0.0"
    debug: bool = False
    pythonpath: str = "."

    # Database
    # FIX: default port corrected to 5432 (docker-compose maps 5432:5432)
    database_url: str = "postgresql+asyncpg://anthology:anthology@localhost:5432/anthology"

    # Redis (optional)
    redis_url: str = "redis://localhost:6379"

    # Paths
    indexes_dir: str = "indexes"
    papers_dir: str = "data/papers"
    registry_path: str = "data/download_registry.json"
    chunks_path: str = "indexes/chunks_metadata.json"  # FIX: was missing, caused paper_service crash

    # Groq
    groq_api_key: str = ""
    use_groq: bool = False
    groq_model: str = "llama-3.1-8b-instant"
    cohere_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://us.cloud.langfuse.com"

    # CORS — FIX: added Vite dev server (5173), removed Streamlit (8501)
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
