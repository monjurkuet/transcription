"""Groq provider adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ...domain.models import TranscriptResult, TranscriptSegment
from ...services.router import ProviderKeyPool
from .base import TranscriptionProvider, coerce_provider_error


class GroqProvider(TranscriptionProvider):
    """Groq transcription provider."""

    provider_name = "groq"

    def __init__(self, key_pool: ProviderKeyPool, model: str, timeout_sec: int):
        self.key_pool = key_pool
        self.model = model
        self.timeout_sec = timeout_sec
        self.url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def transcribe(
        self,
        audio_path: Path,
        content_type: str,
        model_override: Optional[str] = None,
    ) -> TranscriptResult:
        key_state = self.key_pool.acquire()
        if not key_state:
            raise RuntimeError("No Groq keys are currently available")

        try:
            with open(audio_path, "rb") as file_obj:
                response = requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {key_state.raw_key}"},
                    files={"file": (audio_path.name, file_obj, content_type)},
                    data={
                        "model": model_override or self.model,
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "segment",
                    },
                    timeout=self.timeout_sec,
                )
        except requests.exceptions.RequestException as exc:
            message = f"Groq request failed: {exc}"
            self.key_pool.mark_error(key_state.key_id, message)
            raise coerce_provider_error(503, message) from exc

        if response.status_code != 200:
            message = f"Groq transcription failed ({response.status_code}): {response.text}"
            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", "60") or "60")
                self.key_pool.cooldown(key_state.key_id, retry_after, message)
            else:
                self.key_pool.mark_error(key_state.key_id, message)
            raise coerce_provider_error(response.status_code, message)

        payload = response.json()
        segments = [
            TranscriptSegment(
                id=segment.get("id"),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=segment.get("text", ""),
                provider_data={k: v for k, v in segment.items() if k not in {"id", "start", "end", "text", "tokens"}},
            )
            for segment in payload.get("segments", [])
        ]
        return TranscriptResult(
            text=payload.get("text", ""),
            segments=segments,
            provider=self.provider_name,
            model=model_override or self.model,
            raw=payload,
        )

    def status(self) -> Dict[str, Any]:
        return {"provider": self.provider_name, "keys": [item.to_dict() for item in self.key_pool.status()]}
