"""whisper.cpp provider adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ...domain.models import TranscriptResult, TranscriptSegment
from .base import TranscriptionProvider, coerce_provider_error


class WhisperCppProvider(TranscriptionProvider):
    """Local whisper.cpp fallback provider."""

    provider_name = "whisper_cpp"

    def __init__(
        self,
        base_url: str,
        timeout_sec: int,
        temperature: float,
        temperature_inc: float,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.temperature_inc = temperature_inc

    def load_model(self, model_path: str) -> None:
        try:
            response = requests.post(
                f"{self.base_url}/load",
                files={"model": (None, model_path)},
                timeout=self.timeout_sec,
            )
        except requests.exceptions.RequestException as exc:
            raise coerce_provider_error(503, f"whisper.cpp load failed: {exc}") from exc
        if response.status_code >= 400:
            raise coerce_provider_error(response.status_code, f"whisper.cpp load failed: {response.text}")

    def transcribe(
        self,
        audio_path: Path,
        content_type: str,
        model_override: Optional[str] = None,
    ) -> TranscriptResult:
        try:
            with open(audio_path, "rb") as file_obj:
                response = requests.post(
                    f"{self.base_url}/inference",
                    files={"file": (audio_path.name, file_obj, content_type)},
                    data={
                        "temperature": self.temperature,
                        "temperature_inc": self.temperature_inc,
                        "response_format": "json",
                    },
                    timeout=self.timeout_sec,
                )
        except requests.exceptions.RequestException as exc:
            raise coerce_provider_error(503, f"whisper.cpp inference failed: {exc}") from exc
        if response.status_code != 200:
            raise coerce_provider_error(response.status_code, f"whisper.cpp inference failed: {response.text}")

        payload = response.json()
        segments = [
            TranscriptSegment(
                id=segment.get("id"),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=segment.get("text", ""),
                provider_data={k: v for k, v in segment.items() if k not in {"id", "start", "end", "text"}},
            )
            for segment in payload.get("segments", [])
        ]
        return TranscriptResult(
            text=payload.get("text", payload.get("result", "")),
            segments=segments,
            provider=self.provider_name,
            model=model_override,
            raw=payload,
        )

    def status(self) -> Dict[str, Any]:
        return {"provider": self.provider_name, "base_url": self.base_url}
