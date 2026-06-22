from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Anthology API"
    app_version: str = "1.0.0"
    debug: bool = False
    pythonpath: str = "."

    # Database
    # No default provided to ensure it's set via environment variables for security
    database_url: str = ""

    # Redis (optional)
    redis_url: str = ""

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

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "dev", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
