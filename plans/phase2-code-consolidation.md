# Phase 2: Code Consolidation

Priority: MEDIUM | Effort: MEDIUM | Risk if Skipped: Technical debt accumulation, inconsistent behavior

This phase eliminates code duplication and establishes consistent patterns across the codebase.

---

## 2.1 Extract RemoteAPIProvider Base Class

**Files:** 
- `src/audio_transcript/infra/providers/base.py` (modify)
- `src/audio_transcript/infra/providers/groq.py` (modify)
- `src/audio_transcript/infra/providers/mistral.py` (modify)

**Problem:**
Groq and Mistral providers share ~90% identical code. Both implement:
- Key pool acquisition
- HTTP POST with multipart upload
- Error handling with cooldown for 429s
- Response parsing to TranscriptResult

Only differences:
- URL endpoint
- Provider name
- Minor response normalization (Groq excludes `tokens` from provider_data)

**Solution:**
Create `RemoteAPIProvider` abstract base class that encapsulates shared logic.

**Implementation:**

**base.py (updated):**
```python
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


def parse_segments(
    payload: Dict[str, Any],
    excluded_keys: Optional[Set[str]] = None,
) -> List[TranscriptSegment]:
    """Parse segments from provider JSON response.
    
    Args:
        payload: JSON response containing 'segments' array
        excluded_keys: Keys to exclude from provider_data (in addition to standard keys)
    
    Returns:
        List of TranscriptSegment objects
    """
    standard_excluded = {"id", "start", "end", "text"}
    all_excluded = standard_excluded | (excluded_keys or set())
    
    segments = []
    for segment in payload.get("segments", []):
        segments.append(
            TranscriptSegment(
                id=segment.get("id"),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=segment.get("text", ""),
                provider_data={k: v for k, v in segment.items() if k not in all_excluded},
            )
        )
    return segments


class RemoteAPIProvider(TranscriptionProvider):
    """Base class for remote API transcription providers.
    
    Subclasses must define:
        - provider_name: str
        - url: str
        - _excluded_segment_keys(): Set[str] - keys to exclude from provider_data
    
    Subclasses may override:
        - _build_request_data(): customize request payload
        - _extract_text(): customize text extraction from response
    """

    provider_name: str
    url: str

    def __init__(self, key_pool: ProviderKeyPool, model: str, timeout_sec: int):
        self.key_pool = key_pool
        self.model = model
        self.timeout_sec = timeout_sec

    def _excluded_segment_keys(self) -> Set[str]:
        """Keys to exclude from segment provider_data. Override in subclass."""
        return set()

    def _build_request_data(self, model: str) -> Dict[str, Any]:
        """Build the request data payload. Override for custom fields."""
        return {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        """Extract transcript text from response. Override for custom extraction."""
        return payload.get("text", "")

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
        segments = parse_segments(payload, self._excluded_segment_keys())

        return TranscriptResult(
            text=self._extract_text(payload),
            segments=segments,
            provider=self.provider_name,
            model=effective_model,
            raw=payload,
        )

    def status(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "keys": [item.to_dict() for item in self.key_pool.status()],
        }
```

**groq.py (simplified):**
```python
"""Groq provider adapter."""

from __future__ import annotations

from typing import Set

from ...services.router import ProviderKeyPool
from .base import RemoteAPIProvider


class GroqProvider(RemoteAPIProvider):
    """Groq transcription provider."""

    provider_name = "groq"
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, key_pool: ProviderKeyPool, model: str, timeout_sec: int):
        super().__init__(key_pool, model, timeout_sec)

    def _excluded_segment_keys(self) -> Set[str]:
        """Groq includes 'tokens' array we don't want in provider_data."""
        return {"tokens"}
```

**mistral.py (simplified):**
```python
"""Mistral provider adapter."""

from __future__ import annotations

from typing import Set

from ...services.router import ProviderKeyPool
from .base import RemoteAPIProvider


class MistralProvider(RemoteAPIProvider):
    """Mistral transcription provider."""

    provider_name = "mistral"
    url = "https://api.mistral.ai/v1/audio/transcriptions"

    def __init__(self, key_pool: ProviderKeyPool, model: str, timeout_sec: int):
        super().__init__(key_pool, model, timeout_sec)

    def _excluded_segment_keys(self) -> Set[str]:
        """Mistral doesn't have extra keys to exclude."""
        return set()
```

**Lines Removed:**
- groq.py: ~50 lines → ~15 lines
- mistral.py: ~50 lines → ~15 lines
- Total: ~70 lines of duplication eliminated

**Test:**
```python
def test_groq_excludes_tokens_from_provider_data():
    """Verify Groq-specific token exclusion."""
    from audio_transcript.infra.providers.base import parse_segments
    
    payload = {
        "segments": [{
            "id": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "tokens": [1, 2, 3],  # Should be excluded for Groq
            "avg_logprob": -0.5,
        }]
    }
    
    segments = parse_segments(payload, excluded_keys={"tokens"})
    assert "tokens" not in segments[0].provider_data
    assert "avg_logprob" in segments[0].provider_data
```

---

## 2.2 Extract Segment Parsing Helper

**File:** `src/audio_transcript/infra/providers/base.py`

**Problem:**
Segment parsing logic is duplicated across:
- `groq.py` lines 54-61
- `mistral.py` lines 54-61  
- `whisper_cpp.py` lines 53-60

All three parse `segments` array into `TranscriptSegment` with minor variations.

**Solution:**
Already addressed in 2.1 with `parse_segments()` function. Update `whisper_cpp.py` to use it.

**whisper_cpp.py (updated):**
```python
"""whisper.cpp provider adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ...domain.models import TranscriptResult
from .base import TranscriptionProvider, coerce_provider_error, parse_segments


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
        segments = parse_segments(payload)  # Use shared helper

        return TranscriptResult(
            text=payload.get("text", payload.get("result", "")),
            segments=segments,
            provider=self.provider_name,
            model=model_override,
            raw=payload,
        )

    def status(self) -> Dict[str, Any]:
        return {"provider": self.provider_name, "base_url": self.base_url}
```

---

## 2.3 Consolidate Entry Points

**Files:**
- `main.py` (delete)
- `app.py` (keep as single entry point)

**Problem:**
Both `main.py` and `app.py` contain identical code:
```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from audio_transcript.api.app import create_app
application = create_app()
```

This creates confusion about which to use and potential drift.

**Solution:**
Delete `main.py`, use `app.py` as the single WSGI entry point.

**Implementation:**
```bash
# Remove main.py
rm main.py

# Update any references in:
# - README.md
# - Dockerfile (if exists)
# - Procfile (if exists)
# - systemd service files
# - gunicorn/uwsgi configs
```

**Update README.md:**
```markdown
## Running the Server

### Development
```bash
flask --app app:application run --debug
```

### Production
```bash
gunicorn app:application -w 4 -b 0.0.0.0:8000
```
```

---

## 2.4 Normalize provider_data Handling

**Problem:**
Inconsistent handling of segment metadata across providers:
- Groq excludes `tokens` from provider_data
- Mistral keeps everything
- No documented schema for what provider_data should contain

**Solution:**
Define explicit schema and normalize all providers to follow it.

**Implementation:**

**Add to domain/models.py:**
```python
# Standard keys that are always stored as top-level segment fields
SEGMENT_STANDARD_KEYS = frozenset({"id", "start", "end", "text"})

# Keys that should be excluded from provider_data across all providers
SEGMENT_EXCLUDED_KEYS = frozenset({"tokens", "token_ids", "word_timestamps"})

# Keys that represent quality metrics (always preserved)
SEGMENT_QUALITY_KEYS = frozenset({
    "avg_logprob",
    "compression_ratio", 
    "no_speech_prob",
    "temperature",
    "seek",
})
```

**Update parse_segments in base.py:**
```python
from ...domain.models import SEGMENT_EXCLUDED_KEYS, SEGMENT_STANDARD_KEYS


def parse_segments(
    payload: Dict[str, Any],
    additional_excluded_keys: Optional[Set[str]] = None,
) -> List[TranscriptSegment]:
    """Parse segments from provider JSON response.
    
    Standard keys (id, start, end, text) are extracted to segment fields.
    Excluded keys (tokens, etc.) are filtered out.
    Remaining keys are stored in provider_data for quality metrics and debugging.
    
    Args:
        payload: JSON response containing 'segments' array
        additional_excluded_keys: Provider-specific keys to exclude
    
    Returns:
        List of TranscriptSegment objects
    """
    all_excluded = SEGMENT_STANDARD_KEYS | SEGMENT_EXCLUDED_KEYS | (additional_excluded_keys or set())
    
    segments = []
    for segment in payload.get("segments", []):
        segments.append(
            TranscriptSegment(
                id=segment.get("id"),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=segment.get("text", ""),
                provider_data={k: v for k, v in segment.items() if k not in all_excluded},
            )
        )
    return segments
```

**Effect:**
- All providers now have consistent provider_data contents
- Quality metrics (avg_logprob, compression_ratio, no_speech_prob) always preserved
- Large arrays (tokens) always excluded
- Schema is documented and centralized

---

## 2.5 Create Dedicated AuthenticationError

**Files:**
- `src/audio_transcript/domain/errors.py` (modify)
- `src/audio_transcript/api/auth.py` (modify)
- `src/audio_transcript/api/errors.py` (modify)

**Problem:**
Authentication failures use `ValidationError` with a magic string check:
```python
# auth.py
raise ValidationError("Unauthorized")

# errors.py
status = 401 if str(exc) == "Unauthorized" else 400
```

This is fragile - typos break it, and it conflates two different error types.

**Solution:**
Create dedicated `AuthenticationError` class.

**Implementation:**

**domain/errors.py:**
```python
"""Domain and service exceptions."""


class AudioTranscriptError(Exception):
    """Base application error."""


class ConfigurationError(AudioTranscriptError):
    """Raised when required configuration is missing or invalid."""


class AuthenticationError(AudioTranscriptError):
    """Raised when authentication fails (invalid or missing credentials)."""


class AuthorizationError(AudioTranscriptError):
    """Raised when authenticated user lacks permission for an action."""


class ValidationError(AudioTranscriptError):
    """Raised when a request cannot be processed due to invalid input."""


class ArtifactNotFoundError(AudioTranscriptError):
    """Raised when a stored job artifact cannot be located."""


class JobNotFoundError(AudioTranscriptError):
    """Raised when a job id does not exist."""


class RetryableProviderError(AudioTranscriptError):
    """Raised for provider failures that are safe to retry."""


class NonRetryableProviderError(AudioTranscriptError):
    """Raised for provider failures that should not be retried."""
```

**auth.py:**
```python
"""Authentication helpers."""

from __future__ import annotations

import secrets
from functools import wraps
from typing import Callable

from flask import current_app, request

from ..domain.errors import AuthenticationError


def require_api_key(view: Callable):
    """Require the configured inbound API key."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        expected = current_app.config["settings"].service_api_key
        provided = request.headers.get("X-API-Key") or ""
        if not secrets.compare_digest(provided, expected):
            raise AuthenticationError("Invalid or missing API key")
        return view(*args, **kwargs)

    return wrapper
```

**errors.py:**
```python
"""Flask error handlers."""

from __future__ import annotations

import logging

from flask import jsonify

from ..domain.errors import (
    AudioTranscriptError,
    AuthenticationError,
    AuthorizationError,
    JobNotFoundError,
    ValidationError,
)

logger = logging.getLogger("audio_transcript.api.errors")


def register_error_handlers(app) -> None:
    """Register consistent JSON errors."""

    @app.errorhandler(JobNotFoundError)
    def handle_not_found(exc):
        return jsonify({"error": {"code": "job_not_found", "message": str(exc), "details": {}}}), 404

    @app.errorhandler(AuthenticationError)
    def handle_authentication(exc):
        return jsonify({"error": {"code": "unauthorized", "message": "Unauthorized", "details": {}}}), 401

    @app.errorhandler(AuthorizationError)
    def handle_authorization(exc):
        return jsonify({"error": {"code": "forbidden", "message": str(exc), "details": {}}}), 403

    @app.errorhandler(ValidationError)
    def handle_validation(exc):
        return jsonify({"error": {"code": "validation_error", "message": str(exc), "details": {}}}), 400

    @app.errorhandler(AudioTranscriptError)
    def handle_domain(exc):
        logger.error("Domain error: %s", exc, exc_info=True)
        return jsonify({"error": {"code": "application_error", "message": "An application error occurred", "details": {}}}), 500

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        logger.exception("Unexpected error: %s", exc)
        return jsonify({"error": {"code": "internal_error", "message": "An unexpected error occurred", "details": {}}}), 500
```

---

## Verification Checklist

After implementing Phase 2:

- [ ] `pytest tests/` passes (all existing tests)
- [ ] Provider tests verify base class works for both Groq and Mistral
- [ ] Segment parsing is consistent across all providers
- [ ] Only `app.py` exists as entry point
- [ ] Authentication errors return 401 without string matching
- [ ] provider_data contains consistent keys across providers

---

## Files Modified

| File | Change |
|------|--------|
| `src/audio_transcript/infra/providers/base.py` | Add RemoteAPIProvider, parse_segments |
| `src/audio_transcript/infra/providers/groq.py` | Simplify to extend RemoteAPIProvider |
| `src/audio_transcript/infra/providers/mistral.py` | Simplify to extend RemoteAPIProvider |
| `src/audio_transcript/infra/providers/whisper_cpp.py` | Use parse_segments helper |
| `src/audio_transcript/domain/errors.py` | Add AuthenticationError, AuthorizationError |
| `src/audio_transcript/domain/models.py` | Add SEGMENT_*_KEYS constants |
| `src/audio_transcript/api/auth.py` | Use AuthenticationError, secrets.compare_digest |
| `src/audio_transcript/api/errors.py` | Separate handlers for auth vs validation |
| `main.py` | DELETE |
| `README.md` | Update entry point references |

---

## Lines of Code Impact

| Before | After | Savings |
|--------|-------|---------|
| groq.py: 70 lines | 20 lines | -50 lines |
| mistral.py: 70 lines | 20 lines | -50 lines |
| whisper_cpp.py: 70 lines | 55 lines | -15 lines |
| main.py: 8 lines | 0 lines | -8 lines |
| **Total** | | **~123 lines removed** |
