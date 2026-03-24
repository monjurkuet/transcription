# Phase 7: Documentation & Observability

Priority: LOW | Effort: MEDIUM | Risk if Skipped: Onboarding friction, debugging difficulty

This phase improves documentation and adds observability features for production operations.

---

## 7.1 Add Docstrings

**Problem:**
Most public methods lack docstrings, making it hard to understand:
- What parameters are expected
- What exceptions can be raised
- What the return values mean

**Solution:**
Add comprehensive docstrings to all public classes and methods.

**Implementation:**

**Example: services/transcription.py**
```python
"""Transcription orchestration service.

This module contains the TranscriptionService which coordinates the entire
transcription workflow including:
- Job state management
- Provider selection and fallback
- Audio chunking for large files
- Result artifact storage
"""

from __future__ import annotations

import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import Settings
from ..domain.errors import NonRetryableProviderError, RetryableProviderError, ValidationError
from ..domain.models import (
    JobStatus,
    ProviderAttempt,
    TranscriptionJob,
    TranscriptResult,
    is_supported_audio_file,
    utcnow,
)
from ..infra.providers.base import TranscriptionProvider
from ..infra.repository import JobRepository
from ..infra.runtime_state import RuntimeState
from ..infra.storage import TranscriptArtifactStore
from .audio import AudioChunker, AudioInspector, merge_transcripts
from .router import ProviderRouter


@dataclass
class RuntimeDependencies:
    """Container for service dependencies.
    
    Attributes:
        settings: Application configuration
        repository: Job persistence layer
        artifact_store: Transcript storage
        runtime_state: Distributed state (locks, cooldowns)
        router: Provider selection strategy
        remote_providers: Dict of provider name to TranscriptionProvider
        fallback_provider: Optional local fallback (whisper.cpp)
        inspector: Audio metadata extractor
        chunker: Audio splitting utility
        max_parallel_chunks: Concurrent chunk transcriptions (default: 3)
    """
    settings: Settings
    repository: JobRepository
    artifact_store: TranscriptArtifactStore
    runtime_state: RuntimeState
    router: ProviderRouter
    remote_providers: Dict[str, TranscriptionProvider]
    fallback_provider: Optional[TranscriptionProvider]
    inspector: AudioInspector
    chunker: AudioChunker
    max_parallel_chunks: int = 3


class TranscriptionService:
    """Coordinates transcription job execution.
    
    This service handles the complete lifecycle of a transcription job:
    1. Validates the input file
    2. Extracts audio metadata
    3. Chunks large files if necessary
    4. Routes to appropriate provider(s)
    5. Stores results as Parquet artifacts
    
    Thread Safety:
        This service is thread-safe. Multiple workers can process
        different jobs concurrently. Job locks prevent duplicate
        processing of the same job.
    
    Example:
        ```python
        service = TranscriptionService(deps)
        service.create_job(job)  # Persist job
        result = service.process_job(job.job_id)  # Execute transcription
        ```
    
    Attributes:
        deps: Runtime dependencies container
        logger: Service-specific logger
    """

    def __init__(self, deps: RuntimeDependencies):
        """Initialize the transcription service.
        
        Args:
            deps: Container with all required dependencies
        """
        self.deps = deps
        self.logger = logging.getLogger("audio_transcript.transcription")
        self._chunk_executor = ThreadPoolExecutor(
            max_workers=deps.max_parallel_chunks,
            thread_name_prefix="chunk-transcribe-"
        )

    def create_job(self, job: TranscriptionJob) -> None:
        """Persist a new transcription job.
        
        Args:
            job: Job to create with status=QUEUED
        
        Note:
            This only persists the job. Call process_job() to execute it,
            or enqueue it for a worker to process.
        """
        self.deps.repository.create(job)

    def process_job(self, job_id: str) -> TranscriptionJob:
        """Execute a transcription job.
        
        This method:
        1. Acquires a distributed lock to prevent duplicate processing
        2. Validates the source audio file
        3. Extracts audio metadata
        4. Transcribes (with chunking if file is large)
        5. Stores results and updates job status
        
        Args:
            job_id: ID of the job to process
        
        Returns:
            The completed TranscriptionJob with status SUCCEEDED or FAILED
        
        Raises:
            ValidationError: If job is already being processed, or
                source file is missing/invalid
            RetryableProviderError: If all providers failed with retryable errors
            NonRetryableProviderError: If a provider returned a permanent error
        
        Note:
            On any exception, the job status is set to FAILED and the
            error message is recorded.
        """
        # ... implementation ...

    def _transcribe_single(
        self,
        audio_path: Path,
        content_type: str,
        model_override: Optional[str],
        job: TranscriptionJob,
    ) -> TranscriptResult:
        """Transcribe a single audio file through provider chain.
        
        Tries remote providers in order determined by the router,
        falling back to whisper.cpp if all remote providers fail.
        
        Args:
            audio_path: Path to audio file
            content_type: MIME type of the audio
            model_override: Optional specific model to use
            job: Parent job (for recording attempts)
        
        Returns:
            TranscriptResult from the successful provider
        
        Raises:
            NonRetryableProviderError: If a provider returned HTTP 4xx
            RetryableProviderError: If all providers failed with 5xx/timeout
        """
        # ... implementation ...

    def shutdown(self) -> None:
        """Gracefully shutdown the service.
        
        Waits for in-progress chunk transcriptions to complete.
        Call this before application shutdown.
        """
        self._chunk_executor.shutdown(wait=True)
```

**Files requiring docstrings:**

| File | Classes/Functions |
|------|-------------------|
| `services/transcription.py` | `TranscriptionService`, `RuntimeDependencies` |
| `services/router.py` | `ProviderRouter`, `ProviderKeyPool`, `KeyState` |
| `services/audio.py` | `AudioInspector`, `AudioChunker`, `merge_transcripts` |
| `infra/repository.py` | `JobRepository`, `PostgresJobRepository` |
| `infra/queue.py` | `JobQueue`, `RedisJobQueue` |
| `infra/runtime_state.py` | `RuntimeState`, `RedisRuntimeState` |
| `infra/storage.py` | `TranscriptArtifactStore` |
| `infra/providers/base.py` | `TranscriptionProvider`, `RemoteAPIProvider` |
| `domain/models.py` | All dataclasses |
| `config.py` | `Settings` |

---

## 7.2 Enhance README

**File:** `README.md`

**Current state:** Basic usage instructions, missing API documentation and operational details.

**Solution:** Add comprehensive sections.

**Implementation:**

```markdown
# Audio Transcript Service

A production-ready audio transcription service with multi-provider support, automatic fallback, and Parquet-based artifact storage.

## Features

- **Multi-provider support**: Groq, Mistral, and local whisper.cpp
- **Automatic failover**: Seamless fallback between providers
- **Large file handling**: Automatic chunking for long audio files
- **API key pooling**: Round-robin distribution with per-key rate limit tracking
- **Parquet storage**: Efficient, queryable transcript storage
- **Job queue**: Redis-backed async processing
- **PostgreSQL persistence**: Durable job tracking

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Flask API     │────▶│  Redis Queue    │────▶│     Worker      │
│  (HTTP/REST)    │     │   (Job Queue)   │     │  (Background)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                              │
         │              ┌─────────────────┐             │
         └─────────────▶│   PostgreSQL    │◀────────────┘
                        │  (Job Store)    │
                        └─────────────────┘
                                │
                        ┌───────┴───────┐
                        ▼               ▼
                  ┌──────────┐   ┌──────────────┐
                  │  Groq    │   │   Mistral    │
                  │  API     │   │     API      │
                  └──────────┘   └──────────────┘
                        │               │
                        └───────┬───────┘
                                ▼
                        ┌──────────────┐
                        │ whisper.cpp  │
                        │  (Fallback)  │
                        └──────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- ffmpeg (for audio processing)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd audio-transcript

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### Database Setup

```bash
# Create database
createdb audio_transcript

# Tables are auto-created on first run
```

### Running

```bash
# Terminal 1: API server
flask --app app:application run --debug

# Terminal 2: Background worker
python -c "from audio_transcript.worker.runner import run_worker_loop; run_worker_loop()"
```

## API Reference

### Authentication

All endpoints except `/v1/health` require the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:5000/v1/jobs
```

### Endpoints

#### Health Check

```http
GET /v1/health
```

Response:
```json
{
  "status": "ok",
  "postgres": "ok",
  "redis_queue": "ok",
  "redis_state": "ok"
}
```

#### Submit Transcription Job

```http
POST /v1/jobs
Content-Type: multipart/form-data

file: <audio file>
model: (optional) specific model override
chunk_duration_sec: (optional) chunk size for large files
chunk_overlap_sec: (optional) overlap between chunks
```

Response (202 Accepted):
```json
{
  "job": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued",
    "filename": "recording.mp3"
  },
  "links": {
    "status": "/v1/jobs/550e8400-e29b-41d4-a716-446655440000",
    "result": "/v1/jobs/550e8400-e29b-41d4-a716-446655440000/result"
  }
}
```

#### Get Job Status

```http
GET /v1/jobs/{job_id}
```

Response:
```json
{
  "job": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "succeeded",
    "filename": "recording.mp3",
    "created_at": "2024-01-15T10:30:00Z",
    "started_at": "2024-01-15T10:30:01Z",
    "completed_at": "2024-01-15T10:30:15Z",
    "provider": "groq",
    "model": "whisper-large-v3"
  },
  "file_metadata": {
    "duration": 120.5,
    "size_bytes": 1920000,
    "format": "mp3",
    "sample_rate": 44100
  },
  "attempts": [
    {
      "provider": "groq",
      "success": true,
      "latency_ms": 14000
    }
  ]
}
```

#### Get Transcription Result

```http
GET /v1/jobs/{job_id}/result
GET /v1/jobs/{job_id}/result?segment_offset=0&segment_limit=100
```

Response:
```json
{
  "job": { "id": "...", "status": "succeeded", ... },
  "file_metadata": { "duration": 120.5, ... },
  "transcript": {
    "text": "Full transcript text...",
    "provider": "groq",
    "model": "whisper-large-v3",
    "segments": [
      {
        "id": 0,
        "start": 0.0,
        "end": 3.5,
        "text": "Hello and welcome...",
        "provider_data": {
          "avg_logprob": -0.25,
          "compression_ratio": 1.2
        }
      }
    ],
    "total_segments": 150
  },
  "pagination": {
    "offset": 0,
    "limit": 100,
    "total": 150,
    "has_more": true
  }
}
```

#### List Jobs

```http
GET /v1/jobs
GET /v1/jobs?status=succeeded&provider=groq&limit=10
GET /v1/jobs?search=meeting&filename=recording
```

Query Parameters:
- `status`: Filter by job status (queued, processing, succeeded, failed)
- `provider`: Filter by transcription provider
- `filename`: Filter by source filename (partial match)
- `search`: Search in transcript text
- `limit`: Maximum results (default: 50)

#### Provider Status

```http
GET /v1/providers/status
```

Response:
```json
{
  "providers": [
    {
      "provider": "groq",
      "keys": [
        {
          "key_id": "gsk_...abc",
          "status": "available",
          "error_count": 0
        },
        {
          "key_id": "gsk_...def",
          "status": "cooldown",
          "cooldown_until": "2024-01-15T10:35:00Z",
          "reason": "rate limit"
        }
      ]
    },
    {
      "provider": "mistral",
      "keys": [...]
    }
  ]
}
```

### Error Responses

All errors follow this format:
```json
{
  "error": {
    "code": "error_code",
    "message": "Human-readable message",
    "details": {}
  }
}
```

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `validation_error` | 400 | Invalid request parameters |
| `unauthorized` | 401 | Missing/invalid API key |
| `forbidden` | 403 | Not authorized for this resource |
| `job_not_found` | 404 | Job ID doesn't exist |
| `result_unavailable` | 409 | Result not ready (job still processing) |
| `audio_processing_error` | 422 | ffmpeg/ffprobe processing failed |
| `application_error` | 500 | Known application error |
| `internal_error` | 500 | Unexpected server error |

## Configuration

See `.env.example` for all configuration options.

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SERVICE_API_KEY` | API authentication key | `sk-your-secret` |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://user:pass@localhost/db` |
| `GROQ_API_KEYS` | Comma-separated Groq keys | `gsk_key1,gsk_key2` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `MISTRAL_API_KEYS` | - | Comma-separated Mistral keys |
| `MAX_FILE_SIZE_MB` | `25` | Max upload size |
| `CHUNK_DURATION_SEC` | `600` | Chunk size for large files |
| `MAX_PARALLEL_CHUNKS` | `3` | Concurrent transcriptions |
| `REQUEST_TIMEOUT_SEC` | `300` | Provider API timeout |

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install ".[prod]"

# API server
CMD ["gunicorn", "app:application", "-w", "4", "-b", "0.0.0.0:8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - postgres
      - redis

  worker:
    build: .
    command: python -c "from audio_transcript.worker.runner import run_worker_loop; run_worker_loop()"
    env_file: .env
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: audio_transcript
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

## Development

### Running Tests

```bash
# Unit tests only
pytest tests/

# Include integration tests (requires Docker)
pytest -m "" tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Type checking
mypy src/

# Formatting
ruff format src/ tests/
```

## License

MIT
```

---

## 7.3 Add Health Check Depth

**File:** `src/audio_transcript/api/routes.py`

**Problem:**
Current `/health` endpoint only checks database/Redis connectivity, not ffmpeg availability.

**Solution:**
Add optional deep health check that verifies ffmpeg/ffprobe.

**Implementation:**

```python
@bp.get("/health")
def health():
    """Basic health check for load balancers.
    
    Query Parameters:
        deep: If "true", also verify ffmpeg/ffprobe availability
    """
    repo = current_app.config["repository"]
    queue = current_app.config["queue"]
    runtime_state = current_app.config["runtime_state"]
    
    status = {"status": "ok"}
    status.update(repo.healthcheck())
    status.update(queue.healthcheck())
    status.update(runtime_state.healthcheck())
    
    # Optional deep check
    if request.args.get("deep", "").lower() == "true":
        status.update(_check_ffmpeg())
    
    # Set overall status based on components
    if any(v != "ok" for k, v in status.items() if k != "status"):
        status["status"] = "degraded"
    
    return jsonify(status)


def _check_ffmpeg() -> dict:
    """Check ffmpeg and ffprobe availability."""
    import shutil
    import subprocess
    
    result = {}
    
    # Check ffmpeg
    if shutil.which("ffmpeg"):
        try:
            proc = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5,
            )
            result["ffmpeg"] = "ok" if proc.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            result["ffmpeg"] = "timeout"
        except Exception as exc:
            result["ffmpeg"] = f"error: {exc}"
    else:
        result["ffmpeg"] = "not_found"
    
    # Check ffprobe
    if shutil.which("ffprobe"):
        try:
            proc = subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                timeout=5,
            )
            result["ffprobe"] = "ok" if proc.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            result["ffprobe"] = "timeout"
        except Exception as exc:
            result["ffprobe"] = f"error: {exc}"
    else:
        result["ffprobe"] = "not_found"
    
    return result
```

**Usage:**
```bash
# Basic health check (fast, for load balancers)
curl http://localhost:5000/v1/health

# Deep health check (includes ffmpeg verification)
curl http://localhost:5000/v1/health?deep=true
```

**Response:**
```json
{
  "status": "ok",
  "postgres": "ok",
  "redis_queue": "ok",
  "redis_state": "ok",
  "ffmpeg": "ok",
  "ffprobe": "ok"
}
```

---

## 7.4 Add Structured Logging

**Problem:**
Current logging lacks consistent structure, making it hard to:
- Correlate logs across requests
- Filter by job_id or provider
- Aggregate metrics from logs

**Solution:**
Add structured JSON logging with consistent fields.

**Implementation:**

**Create src/audio_transcript/logging_utils.py:**
```python
"""Structured logging utilities."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Context variables for request/job correlation
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_job_id: ContextVar[Optional[str]] = ContextVar("job_id", default=None)


def set_request_context(request_id: str) -> None:
    """Set the current request ID for log correlation."""
    _request_id.set(request_id)


def set_job_context(job_id: str) -> None:
    """Set the current job ID for log correlation."""
    _job_id.set(job_id)


def clear_context() -> None:
    """Clear all context variables."""
    _request_id.set(None)
    _job_id.set(None)


class StructuredFormatter(logging.Formatter):
    """JSON log formatter with consistent fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation IDs
        if request_id := _request_id.get():
            log_data["request_id"] = request_id
        if job_id := _job_id.get():
            log_data["job_id"] = job_id

        # Add extra fields from record
        for key in ("provider", "duration_ms", "status_code", "attempt", "error"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        level = record.levelname[0]  # First letter: I, W, E, D
        
        # Build context string
        context_parts = []
        if request_id := _request_id.get():
            context_parts.append(f"req={request_id[:8]}")
        if job_id := _job_id.get():
            context_parts.append(f"job={job_id[:8]}")
        
        context = f" [{', '.join(context_parts)}]" if context_parts else ""
        
        # Add extra fields
        extras = []
        for key in ("provider", "duration_ms", "status_code"):
            if hasattr(record, key):
                extras.append(f"{key}={getattr(record, key)}")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        
        base = f"{timestamp} {level} {record.name}: {record.getMessage()}{context}{extra_str}"
        
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        
        return base


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, use JSON format; otherwise human-readable
    """
    root = logging.getLogger()
    root.setLevel(level)
    
    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Add new handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(HumanFormatter())
    
    root.addHandler(handler)
    
    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


class LogContext:
    """Context manager for structured log fields."""
    
    def __init__(self, **fields):
        self.fields = fields
        self._old_factory = None
    
    def __enter__(self):
        self._old_factory = logging.getLogRecordFactory()
        fields = self.fields
        
        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            for key, value in fields.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, *args):
        logging.setLogRecordFactory(self._old_factory)
```

**Update services/transcription.py to use structured logging:**
```python
from ..logging_utils import LogContext, set_job_context, clear_context


class TranscriptionService:
    def process_job(self, job_id: str) -> TranscriptionJob:
        set_job_context(job_id)
        try:
            # ... existing code ...
            
            self.logger.info(
                "Job processing started",
                extra={"provider": "pending", "duration_ms": 0}
            )
            
            # ... transcribe ...
            
            self.logger.info(
                "Job completed successfully",
                extra={
                    "provider": transcript.provider,
                    "duration_ms": int((time.monotonic() - start) * 1000),
                    "segment_count": len(transcript.segments),
                }
            )
        finally:
            clear_context()

    def _run_provider(self, provider, ...):
        with LogContext(provider=provider.provider_name):
            start = time.monotonic()
            try:
                result = provider.transcribe(...)
                self.logger.info(
                    "Provider transcription succeeded",
                    extra={"duration_ms": int((time.monotonic() - start) * 1000)}
                )
                return result
            except Exception as exc:
                self.logger.warning(
                    "Provider transcription failed",
                    extra={
                        "duration_ms": int((time.monotonic() - start) * 1000),
                        "error": str(exc),
                    }
                )
                raise
```

**Update api/app.py to add request context:**
```python
import uuid
from ..logging_utils import configure_logging, set_request_context, clear_context


def create_app(settings: Settings = None, ...):
    # ... existing code ...
    
    configure_logging(
        level=settings.log_level,
        json_format=os.getenv("LOG_FORMAT", "").lower() == "json",
    )
    
    @app.before_request
    def set_request_id():
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_context(request_id)
        g.request_id = request_id
    
    @app.after_request
    def add_request_id_header(response):
        if hasattr(g, "request_id"):
            response.headers["X-Request-ID"] = g.request_id
        return response
    
    @app.teardown_request
    def clear_request_context(exception=None):
        clear_context()
```

**Example JSON log output:**
```json
{"timestamp": "2024-01-15T10:30:15.123Z", "level": "INFO", "logger": "audio_transcript.transcription", "message": "Job completed successfully", "request_id": "abc123", "job_id": "550e8400", "provider": "groq", "duration_ms": 14523, "segment_count": 45}
```

**Example human log output:**
```
10:30:15 I audio_transcript.transcription: Job completed successfully [req=abc123, job=550e8400] (provider=groq, duration_ms=14523)
```

---

## Verification Checklist

After implementing Phase 7:

- [ ] All public classes/methods have docstrings
- [ ] README covers all API endpoints with examples
- [ ] README includes deployment instructions
- [ ] `/health?deep=true` verifies ffmpeg availability
- [ ] Logs include job_id and request_id
- [ ] JSON log format works with `LOG_FORMAT=json`
- [ ] Log aggregation queries work (filter by job_id, provider)

---

## Files Modified/Created

| File | Change |
|------|--------|
| `src/audio_transcript/services/transcription.py` | Add docstrings, structured logging |
| `src/audio_transcript/services/router.py` | Add docstrings |
| `src/audio_transcript/services/audio.py` | Add docstrings |
| `src/audio_transcript/infra/repository.py` | Add docstrings |
| `src/audio_transcript/infra/queue.py` | Add docstrings |
| `src/audio_transcript/infra/storage.py` | Add docstrings |
| `src/audio_transcript/api/routes.py` | Add deep health check |
| `src/audio_transcript/api/app.py` | Add request context logging |
| `src/audio_transcript/logging_utils.py` | CREATE - structured logging |
| `src/audio_transcript/config.py` | Add docstrings |
| `README.md` | Complete rewrite with API docs |

---

## Configuration Changes

Add to `.env.example`:
```bash
# Logging
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=             # Set to "json" for structured JSON logs
```

---

## Monitoring Integration

The structured logs are compatible with:

- **Datadog**: JSON logs auto-parsed
- **ELK Stack**: Filter by `job_id`, `provider`, `duration_ms`
- **CloudWatch Logs Insights**: Query by fields
- **Grafana Loki**: Label extraction from JSON

Example Loki query:
```logql
{app="audio-transcript"} | json | provider="groq" | duration_ms > 10000
```

Example CloudWatch Insights query:
```sql
fields @timestamp, message, job_id, provider, duration_ms
| filter level = "ERROR"
| sort @timestamp desc
| limit 100
```
