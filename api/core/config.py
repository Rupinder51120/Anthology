from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Anthology API"
    app_version: str = "1.0.0"
    debug: bool = False
    pythonpath: str = "."

    # Database
    database_url: str = "postgresql+asyncpg://anthology:anthology@localhost:5433/anthology"

    # Redis (optional)
    redis_url: str = "redis://localhost:6379"

    # Paths
    indexes_dir: str = "indexes"
    papers_dir: str = "data/papers"
    chunks_path: str = "indexes/chunks_metadata.json"
    embeddings_path: str = "indexes/chunk_embeddings.npy"
    registry_path: str = "data/download_registry.json"

    # Groq
    groq_api_key: str = ""
    use_groq: bool = False
    groq_model: str = "llama-3.1-8b-instant"

    # pgvector
    use_pgvector: bool = False

    # CORS
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
