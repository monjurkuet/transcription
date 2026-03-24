"""Provider interfaces and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from ...domain.errors import NonRetryableProviderError, RetryableProviderError
from ...domain.models import TranscriptResult


class TranscriptionProvider(ABC):
    """Provider adapter interface."""

    provider_name: str

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        content_type: str,
        model_override: Optional[str] = None,
    ) -> TranscriptResult:
        """Transcribe an audio artifact and return a normalized transcript."""

    def load_model(self, model_path: str) -> None:
        """Optional startup hook for providers that support loading."""

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Expose runtime provider status."""


def coerce_provider_error(status_code: int, message: str) -> Exception:
    """Map provider HTTP failures to retry behavior."""
    if status_code == 429 or status_code >= 500:
        return RetryableProviderError(message)
    if status_code in {400, 401, 403, 404, 422}:
        return NonRetryableProviderError(message)
    return RetryableProviderError(message)
