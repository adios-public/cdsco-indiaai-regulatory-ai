"""Local LLM client via Ollama API — no external API key required.

All inference stays on-device (AIKosh / Ollama), making the solution
fully sovereign and DPDP Act 2023 compliant.
"""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send a chat completion request to the local Ollama endpoint.

    Args:
        system: System prompt.
        user:   User message.
        model:  Ollama model name. Defaults to settings.default_model.
                Pass settings.powerful_model for summarisation / long reports.
        max_tokens: Override for max response tokens.

    Returns:
        The model's text response.
    """
    settings = get_settings()
    target_model = model or settings.default_model

    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {
            "num_predict": max_tokens or settings.llm_max_tokens,
            "temperature": 0.1,   # low temp for deterministic regulatory outputs
        },
    }

    response = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json=payload,
        timeout=settings.llm_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def complete_powerful(system: str, user: str, max_tokens: int | None = None) -> str:
    """Use the more capable model (qwen3.6) for complex long-form tasks."""
    return complete(system, user, model=get_settings().powerful_model, max_tokens=max_tokens)
