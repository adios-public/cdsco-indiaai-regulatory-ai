from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Ollama / AIKosh local inference
    ollama_base_url: str = "http://localhost:11434"

    # Model routing:
    #   default_model   — used for classification, completeness, inspection
    #   powerful_model  — used for summarisation and long-form report generation
    default_model: str = "gajendra:latest"
    powerful_model: str = "qwen3.6:latest"

    llm_timeout_seconds: int = 120
    llm_max_tokens: int = 4096

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    environment: str = "development"

    anonymisation_salt: str = "default-change-in-production"
    pseudo_token_prefix: str = "TOK"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
