# Phase 1: Critical Security & Reliability Fixes

Priority: HIGH | Effort: LOW-MEDIUM | Risk if Skipped: Security vulnerabilities, data loss

This phase addresses immediate security and reliability concerns that could expose the system to attacks or cause data loss.

---

## 1.1 Timing-Safe API Key Comparison

**File:** `src/audio_transcript/api/auth.py`

**Problem:**
Line 18 uses `!=` for API key comparison:
```python
if request.headers.get("X-API-Key") != expected:
```
This is vulnerable to timing attacks where an attacker can measure response times to deduce valid API keys character-by-character.

**Solution:**
Use `secrets.compare_digest()` which performs constant-time comparison.

**Implementation:**
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
            raise AuthenticationError("Unauthorized")
        return view(*args, **kwargs)

    return wrapper
```

**Dependencies:**
- Requires adding `AuthenticationError` to `domain/errors.py` (see 1.5)

**Test:**
```python
def test_auth_rejects_wrong_key_timing_safe(tmp_path):
    """Verify timing-safe comparison is used."""
    app, _ = build_test_app(tmp_path)
    client = app.test_client()
    # Should reject with 401, not leak timing info
    response = client.get("/v1/providers/status", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401
```

---

## 1.2 File Size Validation at Upload Time

**File:** `src/audio_transcript/api/routes.py`

**Problem:**
File uploads are written to disk without size validation (line 47). The `max_file_size_mb` check only happens later during transcription in `transcription.py`. An attacker could upload arbitrarily large files, exhausting disk space.

**Solution:**
Check `request.content_length` before saving the file.

**Implementation:**
```python
@bp.post("/jobs")
@require_api_key
def create_job():
    settings = current_app.config["settings"]
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    
    # Check Content-Length header first (fast rejection)
    if request.content_length and request.content_length > max_bytes:
        return jsonify({
            "error": {
                "code": "validation_error",
                "message": f"File exceeds maximum size of {settings.max_file_size_mb}MB",
                "details": {"max_bytes": max_bytes, "provided_bytes": request.content_length}
            }
        }), 400

    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": {"code": "validation_error", "message": "file is required", "details": {}}}), 400
    
    # Validate actual file size (handles chunked uploads without Content-Length)
    uploaded.seek(0, 2)  # Seek to end
    actual_size = uploaded.tell()
    uploaded.seek(0)  # Reset to beginning
    
    if actual_size > max_bytes:
        return jsonify({
            "error": {
                "code": "validation_error",
                "message": f"File exceeds maximum size of {settings.max_file_size_mb}MB",
                "details": {"max_bytes": max_bytes, "actual_bytes": actual_size}
            }
        }), 400
    
    if not is_supported_audio_file(Path(uploaded.filename or "")):
        return (
            jsonify({"error": {"code": "validation_error", "message": "unsupported audio file", "details": {}}}),
            400,
        )
    # ... rest of function unchanged
```

**Test:**
```python
def test_api_rejects_oversized_file(tmp_path):
    app, _ = build_test_app(tmp_path)
    client = app.test_client()
    # Create file larger than 25MB default
    large_file = io.BytesIO(b"x" * (26 * 1024 * 1024))
    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "secret"},
        data={"file": (large_file, "large.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "exceeds maximum size" in response.get_json()["error"]["message"]
```

---

## 1.3 Database Connection Pooling

**File:** `src/audio_transcript/infra/repository.py`

**Problem:**
Each database operation creates a new connection via `_connect()` (line 52). This causes:
- Connection overhead on every request
- Potential connection exhaustion under load
- Slower response times

**Solution:**
Use `psycopg_pool.ConnectionPool` for connection reuse.

**Implementation:**
```python
"""Job repository implementations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ..domain.errors import JobNotFoundError
from ..domain.models import FileMetadata, JobPayload, JobStatus, ProviderAttempt, TranscriptionJob


class PostgresJobRepository(JobRepository):
    """Postgres-backed durable job repository."""

    def __init__(self, database_url: str, min_size: int = 2, max_size: int = 10):
        self.database_url = database_url
        self._pool = ConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
        )
        self._ensure_schema()

    @contextmanager
    def _get_connection(self) -> Generator[psycopg.Connection, None, None]:
        """Get a connection from the pool."""
        with self._pool.connection() as conn:
            yield conn

    def _ensure_schema(self) -> None:
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute("""...""")  # existing schema SQL

    def get(self, job_id: str) -> TranscriptionJob:
        with self._get_connection() as conn, conn.cursor() as cur:
            # ... existing implementation using _get_connection instead of _connect

    def close(self) -> None:
        """Close the connection pool. Call during shutdown."""
        self._pool.close()

    def healthcheck(self) -> Dict[str, str]:
        with self._get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        pool_stats = self._pool.get_stats()
        return {
            "postgres": "ok",
            "pool_size": pool_stats.pool_size,
            "pool_available": pool_stats.pool_available,
        }
```

**Dependencies:**
- Add `psycopg_pool` to `pyproject.toml`:
```toml
dependencies = [
    # ... existing
    "psycopg[pool]>=3.1.0",
]
```

**Migration Steps:**
1. Replace all `self._connect()` calls with `self._get_connection()`
2. Add pool parameters to `PostgresJobRepository.__init__`
3. Add `close()` method for graceful shutdown
4. Update `build_runtime()` to configure pool sizes from settings
5. Add pool stats to healthcheck response

---

## 1.4 Dead Letter Queue & Retry with Backoff

**File:** `src/audio_transcript/worker/runner.py`

**Problem:**
Failed jobs are only logged and then lost (lines 24-27):
```python
except Exception:
    logger.exception("job %s failed", job_id)
    time.sleep(1)
```
No retry mechanism, no dead letter queue, jobs are permanently lost.

**Solution:**
Implement retry counter with exponential backoff, move to DLQ after max retries.

**Implementation:**
```python
"""Background worker entrypoint."""

from __future__ import annotations

import logging
import time
from typing import Optional

from ..api.app import build_runtime
from ..config import Settings
from ..domain.models import JobStatus


MAX_RETRIES = 3
BASE_BACKOFF_SEC = 2


def calculate_backoff(attempt: int) -> float:
    """Exponential backoff: 2, 4, 8 seconds."""
    return BASE_BACKOFF_SEC * (2 ** (attempt - 1))


def run_worker_loop(settings: Settings | None = None, poll_interval_sec: int = 1) -> None:
    """Run the background job worker forever."""
    settings = settings or Settings.from_env()
    runtime = build_runtime(settings)
    logger = logging.getLogger("audio_transcript.worker")
    queue = runtime["queue"]
    service = runtime["service"]
    repository = runtime["repository"]

    logger.info("worker started")
    while True:
        job_id = queue.dequeue(timeout=poll_interval_sec)
        if not job_id:
            continue

        retry_count = _get_retry_count(repository, job_id)
        
        try:
            service.process_job(job_id)
        except Exception as exc:
            retry_count += 1
            logger.exception("job %s failed (attempt %d/%d)", job_id, retry_count, MAX_RETRIES)
            
            if retry_count >= MAX_RETRIES:
                # Move to dead letter queue
                logger.error("job %s exceeded max retries, moving to DLQ", job_id)
                queue.move_to_dlq(job_id, str(exc))
                _mark_job_failed_permanent(repository, job_id, f"Max retries exceeded: {exc}")
            else:
                # Requeue with backoff
                backoff = calculate_backoff(retry_count)
                logger.info("job %s will retry after %.1fs backoff", job_id, backoff)
                time.sleep(backoff)
                queue.requeue(job_id, retry_count)


def _get_retry_count(repository, job_id: str) -> int:
    """Get current retry count from job metadata."""
    try:
        job = repository.get(job_id)
        return len([a for a in job.attempts if not a.success])
    except Exception:
        return 0


def _mark_job_failed_permanent(repository, job_id: str, error: str) -> None:
    """Mark job as permanently failed."""
    try:
        job = repository.get(job_id)
        job.status = JobStatus.FAILED
        job.error = error
        repository.save(job)
    except Exception:
        pass  # Best effort
```

**Queue Interface Changes:**
Add to `src/audio_transcript/infra/queue.py`:
```python
class JobQueue(ABC):
    # ... existing methods
    
    @abstractmethod
    def requeue(self, job_id: str, retry_count: int) -> None:
        """Requeue a job for retry."""
    
    @abstractmethod
    def move_to_dlq(self, job_id: str, error: str) -> None:
        """Move failed job to dead letter queue."""
    
    @abstractmethod
    def get_dlq_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List jobs in dead letter queue."""
```

**Redis Implementation:**
```python
class RedisJobQueue(JobQueue):
    def __init__(self, redis_url: str, queue_name: str):
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.dlq_name = f"{queue_name}:dlq"
        self._client = redis.from_url(redis_url)
    
    def requeue(self, job_id: str, retry_count: int) -> None:
        # Store retry count in hash, push back to queue
        self._client.hset(f"{self.queue_name}:retries", job_id, retry_count)
        self._client.lpush(self.queue_name, job_id)
    
    def move_to_dlq(self, job_id: str, error: str) -> None:
        payload = json.dumps({"job_id": job_id, "error": error, "moved_at": utcnow().isoformat()})
        self._client.lpush(self.dlq_name, payload)
        self._client.hdel(f"{self.queue_name}:retries", job_id)
    
    def get_dlq_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        items = self._client.lrange(self.dlq_name, 0, limit - 1)
        return [json.loads(item) for item in items]
```

**Test:**
```python
def test_worker_retries_failed_job(tmp_path):
    """Verify retry with backoff on failure."""
    # Setup with provider that fails twice then succeeds
    fail_count = [0]
    class FlakeyProvider:
        provider_name = "flakey"
        def transcribe(self, *args, **kwargs):
            fail_count[0] += 1
            if fail_count[0] < 3:
                raise RetryableProviderError("temporary failure")
            return TranscriptResult(text="ok", segments=[], provider="flakey")
    # ... test that job eventually succeeds after retries
```

---

## 1.5 Suppress Internal Error Details in Responses

**File:** `src/audio_transcript/api/errors.py`

**Problem:**
Line 27 exposes internal exception messages to clients:
```python
return jsonify({"error": {"code": "internal_error", "message": str(exc), "details": {}}}), 500
```
This could leak file paths, stack traces, or implementation details.

**Solution:**
Log the full exception but return a generic message to clients.

**Implementation:**
```python
"""Flask error handlers."""

from __future__ import annotations

import logging

from flask import jsonify

from ..domain.errors import (
    AudioTranscriptError,
    AuthenticationError,
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
    def handle_auth(exc):
        return jsonify({"error": {"code": "unauthorized", "message": "Unauthorized", "details": {}}}), 401

    @app.errorhandler(ValidationError)
    def handle_validation(exc):
        return jsonify({"error": {"code": "validation_error", "message": str(exc), "details": {}}}), 400

    @app.errorhandler(AudioTranscriptError)
    def handle_domain(exc):
        # Log full error for debugging, return generic message
        logger.error("Domain error: %s", exc, exc_info=True)
        return jsonify({"error": {"code": "application_error", "message": "An application error occurred", "details": {}}}), 500

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        # Log full exception with stack trace
        logger.exception("Unexpected error: %s", exc)
        # Return generic message - never expose internal details
        return jsonify({"error": {"code": "internal_error", "message": "An unexpected error occurred", "details": {}}}), 500
```

**Add AuthenticationError to domain/errors.py:**
```python
"""Domain and service exceptions."""


class AudioTranscriptError(Exception):
    """Base application error."""


class ConfigurationError(AudioTranscriptError):
    """Raised when required configuration is missing or invalid."""


class AuthenticationError(AudioTranscriptError):
    """Raised when authentication fails."""


class ValidationError(AudioTranscriptError):
    """Raised when a request cannot be processed."""


class ArtifactNotFoundError(AudioTranscriptError):
    """Raised when a stored job artifact cannot be located."""


class JobNotFoundError(AudioTranscriptError):
    """Raised when a job id does not exist."""


class RetryableProviderError(AudioTranscriptError):
    """Raised for provider failures that are safe to retry."""


class NonRetryableProviderError(AudioTranscriptError):
    """Raised for provider failures that should not be retried."""
```

**Test:**
```python
def test_internal_errors_do_not_leak_details(tmp_path):
    """Verify 500 errors don't expose internal paths or stack traces."""
    app, runtime = build_test_app(tmp_path)
    
    # Force an internal error by corrupting job state
    job_id = "test-job"
    # ... setup that causes internal error
    
    client = app.test_client()
    response = client.get(f"/v1/jobs/{job_id}/result", headers={"X-API-Key": "secret"})
    
    error_message = response.get_json()["error"]["message"]
    assert "/home/" not in error_message
    assert "Traceback" not in error_message
    assert error_message in ["An unexpected error occurred", "An application error occurred"]
```

---

## Verification Checklist

After implementing Phase 1:

- [ ] `pytest tests/test_api.py` passes
- [ ] Manual test: invalid API key returns 401 without timing leak
- [ ] Manual test: upload 30MB file returns 400 error before disk write
- [ ] Manual test: concurrent requests don't exhaust DB connections
- [ ] Manual test: failing job retries 3 times then appears in DLQ
- [ ] Manual test: 500 errors don't expose file paths in response
- [ ] Check logs contain full error details for debugging
- [ ] Load test: 50 concurrent requests don't create connection issues

---

## Files Modified

| File | Change |
|------|--------|
| `src/audio_transcript/api/auth.py` | Timing-safe comparison, use AuthenticationError |
| `src/audio_transcript/api/routes.py` | Add file size validation |
| `src/audio_transcript/api/errors.py` | Suppress internal details, add AuthenticationError handler |
| `src/audio_transcript/domain/errors.py` | Add AuthenticationError class |
| `src/audio_transcript/infra/repository.py` | Add connection pooling |
| `src/audio_transcript/infra/queue.py` | Add requeue/DLQ methods |
| `src/audio_transcript/worker/runner.py` | Add retry logic with backoff |
| `pyproject.toml` | Add psycopg[pool] dependency |
