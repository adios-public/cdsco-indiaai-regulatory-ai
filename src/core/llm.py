"""Thin wrapper around the Anthropic client with retry logic."""
from __future__ import annotations

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from src.core.config import get_settings


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def complete(system: str, user: str, max_tokens: int | None = None) -> str:
    settings = get_settings()
    response = _client().messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens or settings.llm_max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text
