"""Mistral provider adapter."""

from __future__ import annotations

from ...services.router import ProviderKeyPool
from .base import RemoteAPIProvider


class MistralProvider(RemoteAPIProvider):
    """Mistral transcription provider.

    Uses Mistral's `/v1/audio/transcriptions` HTTP endpoint with multipart uploads.
    """

    provider_name = "mistral"
    url = "https://api.mistral.ai/v1/audio/transcriptions"

    def __init__(self, key_pool: ProviderKeyPool, model: str, timeout_sec: int):
        super().__init__(key_pool, model, timeout_sec)
