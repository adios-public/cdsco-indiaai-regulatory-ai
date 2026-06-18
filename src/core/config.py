from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
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
