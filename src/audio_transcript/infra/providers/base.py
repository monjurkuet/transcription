"""Provider interfaces and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

from ...domain.errors import NonRetryableProviderError, RetryableProviderError
from ...domain.models import TranscriptResult, TranscriptSegment
from ...services.router import ProviderKeyPool


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


def parse_segments(payload: Dict[str, Any], excluded_keys: Optional[Set[str]] = None) -> List[TranscriptSegment]:
    """Normalize provider response segments."""
    excluded = {"id", "start", "end", "text"} | (excluded_keys or set())
    return [
        TranscriptSegment(
            id=segment.get("id"),
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=segment.get("text", ""),
            provider_data={key: value for key, value in segment.items() if key not in excluded},
        )
        for segment in payload.get("segments", [])
    ]


class RemoteAPIProvider(TranscriptionProvider):
    """Shared implementation for remote HTTP transcription providers."""

    provider_name: str
    url: str

    def __init__(self, key_pool: ProviderKeyPool, model: str, timeout_sec: int):
        self.key_pool = key_pool
        self.model = model
        self.timeout_sec = timeout_sec

    def _build_request_data(self, model: str) -> Dict[str, Any]:
        return {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        return payload.get("text", "")

    def _excluded_segment_keys(self) -> Set[str]:
        return set()

    def transcribe(
        self,
        audio_path: Path,
        content_type: str,
        model_override: Optional[str] = None,
    ) -> TranscriptResult:
        key_state = self.key_pool.acquire()
        if not key_state:
            raise RuntimeError(f"No {self.provider_name} keys are currently available")

        effective_model = model_override or self.model

        try:
            with open(audio_path, "rb") as file_obj:
                response = requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {key_state.raw_key}"},
                    files={"file": (audio_path.name, file_obj, content_type)},
                    data=self._build_request_data(effective_model),
                    timeout=self.timeout_sec,
                )
        except requests.exceptions.RequestException as exc:
            message = f"{self.provider_name} request failed: {exc}"
            self.key_pool.mark_error(key_state.key_id, message)
            raise coerce_provider_error(503, message) from exc

        if response.status_code != 200:
            message = f"{self.provider_name} transcription failed ({response.status_code}): {response.text}"
            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", "60") or "60")
                self.key_pool.cooldown(key_state.key_id, retry_after, message)
            else:
                self.key_pool.mark_error(key_state.key_id, message)
            raise coerce_provider_error(response.status_code, message)

        payload = response.json()
        return TranscriptResult(
            text=self._extract_text(payload),
            segments=parse_segments(payload, excluded_keys=self._excluded_segment_keys()),
            provider=self.provider_name,
            model=effective_model,
            raw=payload,
        )

    def status(self) -> Dict[str, Any]:
        return {"provider": self.provider_name, "keys": [item.to_dict() for item in self.key_pool.status()]}
