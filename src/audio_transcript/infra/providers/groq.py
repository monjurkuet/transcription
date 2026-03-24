"""Groq provider adapter."""

from __future__ import annotations

from ...services.router import ProviderKeyPool
from .base import RemoteAPIProvider


class GroqProvider(RemoteAPIProvider):
    """Groq transcription provider."""

    provider_name = "groq"
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, key_pool: ProviderKeyPool, model: str, timeout_sec: int):
        super().__init__(key_pool, model, timeout_sec)

    def _excluded_segment_keys(self) -> set[str]:
        return {"tokens"}
