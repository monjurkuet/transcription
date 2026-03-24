# Phase 5: Configuration & Dependencies

Priority: LOW-MEDIUM | Effort: LOW | Risk if Skipped: Deployment issues, confusion

This phase cleans up configuration handling and dependency management.

---

## 5.1 Consolidate Dependency Files

**Files:**
- `requirements.txt` (delete)
- `pyproject.toml` (keep as single source)

**Problem:**
Both `requirements.txt` and `pyproject.toml` define dependencies:

**requirements.txt:**
```
flask>=3.0.0
psycopg[binary]>=3.1.0
redis>=5.0.0
pyarrow>=14.0.0
python-dotenv>=1.0.0
requests>=2.31.0
werkzeug>=3.0.0
gunicorn>=21.0.0
pytest>=8.0.0
```

**pyproject.toml:**
```toml
dependencies = [
    "flask>=3.0.0",
    "psycopg[binary]>=3.1.0",
    # ... same list
]
```

This creates:
- Maintenance burden (update in two places)
- Risk of drift between files
- Confusion about which is authoritative

**Solution:**
Delete `requirements.txt`, use only `pyproject.toml`. Add pip-tools workflow for pinned requirements if needed.

**Implementation:**

**Step 1: Update pyproject.toml with complete dependencies:**
```toml
[project]
name = "audio-transcript"
version = "0.1.0"
description = "Audio transcription service with provider fallback"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "flask>=3.0.0",
    "psycopg[binary,pool]>=3.1.0",  # Added pool extra
    "redis>=5.0.0",
    "pyarrow>=14.0.0",
    "python-dotenv>=1.0.0",
    "requests>=2.31.0",
    "werkzeug>=3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.12.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]
prod = [
    "gunicorn>=21.0.0",
]

[project.scripts]
audio-transcript-worker = "audio_transcript.worker.runner:run_worker_loop"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
```

**Step 2: Delete requirements.txt:**
```bash
rm requirements.txt
```

**Step 3: Update README.md installation instructions:**
```markdown
## Installation

### Development
```bash
pip install -e ".[dev]"
```

### Production
```bash
pip install ".[prod]"
```

### With pinned versions (optional)
For reproducible builds, generate a locked requirements file:
```bash
pip install pip-tools
pip-compile pyproject.toml -o requirements.lock
pip install -r requirements.lock
```
```

**Step 4: Add .gitignore entry:**
```
# Generated lock files (optional, may want to commit for CI)
requirements.lock
```

---

## 5.2 Add URL Validation

**File:** `src/audio_transcript/config.py`

**Problem:**
URLs are accepted without validation:
```python
redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
database_url=database_url,
whisper_cpp_base_url=os.getenv("WHISPER_CPP_BASE_URL", "http://127.0.0.1:8334"),
```

Invalid URLs cause confusing errors at runtime instead of clear startup failures.

**Solution:**
Validate URL formats during `Settings.from_env()`.

**Implementation:**
```python
"""Configuration loading."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

from .domain.errors import ConfigurationError


def _parse_csv_env(name: str) -> List[str]:
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer, got: {value}")


def _parse_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        raise ConfigurationError(f"{name} must be a number, got: {value}")


def _validate_url(value: str, name: str, allowed_schemes: List[str]) -> str:
    """Validate URL format and scheme.
    
    Args:
        value: URL string to validate
        name: Config variable name (for error messages)
        allowed_schemes: List of allowed URL schemes
    
    Returns:
        The validated URL string
        
    Raises:
        ConfigurationError: If URL is invalid
    """
    if not value:
        raise ConfigurationError(f"{name} is required")
    
    try:
        parsed = urlparse(value)
    except Exception as exc:
        raise ConfigurationError(f"{name} is not a valid URL: {value}") from exc
    
    if not parsed.scheme:
        raise ConfigurationError(f"{name} must include a scheme (e.g., {allowed_schemes[0]}://...): {value}")
    
    if parsed.scheme not in allowed_schemes:
        raise ConfigurationError(
            f"{name} has invalid scheme '{parsed.scheme}'. Allowed: {', '.join(allowed_schemes)}"
        )
    
    if not parsed.netloc and parsed.scheme not in ("file",):
        raise ConfigurationError(f"{name} must include a host: {value}")
    
    return value


def _validate_redis_url(value: str, name: str = "REDIS_URL") -> str:
    """Validate Redis connection URL."""
    return _validate_url(value, name, ["redis", "rediss", "unix"])


def _validate_postgres_url(value: str, name: str = "DATABASE_URL") -> str:
    """Validate PostgreSQL connection URL."""
    return _validate_url(value, name, ["postgresql", "postgres"])


def _validate_http_url(value: str, name: str) -> str:
    """Validate HTTP/HTTPS URL."""
    return _validate_url(value, name, ["http", "https"])


@dataclass
class Settings:
    """Runtime settings."""

    service_api_key: str
    database_url: str
    redis_url: str
    storage_root: Path
    transcript_dataset_root: Path
    groq_api_keys: List[str]
    mistral_api_keys: List[str]
    groq_model: str = "whisper-large-v3"
    mistral_model: str = "voxtral-mini-latest"
    whisper_cpp_base_url: str = "http://127.0.0.1:8334"
    whisper_cpp_model_path: Optional[str] = None
    whisper_cpp_temperature: float = 0.0
    whisper_cpp_temperature_inc: float = 0.2
    request_timeout_sec: int = 300
    provider_max_retries: int = 3
    chunk_duration_sec: int = 600
    chunk_overlap_sec: int = 5
    max_file_size_mb: int = 25
    max_parallel_chunks: int = 3
    log_level: str = "INFO"
    job_retention_days: int = 7
    queue_name: str = "audio-transcript:jobs"
    dynamic_whisper_cpp_load: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from env and validate them.
        
        Raises:
            ConfigurationError: If required settings are missing or invalid
        """
        load_dotenv()
        
        # Required string settings
        service_api_key = os.getenv("SERVICE_API_KEY", "").strip()
        if not service_api_key:
            raise ConfigurationError("SERVICE_API_KEY is required")
        
        # Required URL settings with validation
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise ConfigurationError("DATABASE_URL is required")
        _validate_postgres_url(database_url, "DATABASE_URL")
        
        redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        _validate_redis_url(redis_url, "REDIS_URL")
        
        whisper_cpp_base_url = os.getenv("WHISPER_CPP_BASE_URL", "http://127.0.0.1:8334").rstrip("/")
        _validate_http_url(whisper_cpp_base_url, "WHISPER_CPP_BASE_URL")
        
        # Path settings
        storage_root = Path(os.getenv("STORAGE_ROOT", "./data")).resolve()
        transcript_dataset_root = Path(os.getenv("TRANSCRIPT_DATASET_ROOT", "./transcript_dataset")).resolve()
        
        # API keys
        groq_api_keys = _parse_csv_env("GROQ_API_KEYS")
        if not groq_api_keys:
            single_groq = os.getenv("GROQ_API_KEY", "").strip()
            if single_groq:
                groq_api_keys = [single_groq]

        mistral_api_keys = _parse_csv_env("MISTRAL_API_KEYS")
        if not groq_api_keys and not mistral_api_keys:
            raise ConfigurationError("At least one remote provider key is required (GROQ_API_KEYS or MISTRAL_API_KEYS)")

        # Numeric settings with validation
        chunk_duration = _parse_int("CHUNK_DURATION_SEC", 600)
        chunk_overlap = _parse_int("CHUNK_OVERLAP_SEC", 5)
        if chunk_overlap >= chunk_duration:
            raise ConfigurationError(
                f"CHUNK_OVERLAP_SEC ({chunk_overlap}) must be less than CHUNK_DURATION_SEC ({chunk_duration})"
            )
        
        max_file_size = _parse_int("MAX_FILE_SIZE_MB", 25)
        if max_file_size <= 0:
            raise ConfigurationError(f"MAX_FILE_SIZE_MB must be positive, got: {max_file_size}")

        return cls(
            service_api_key=service_api_key,
            database_url=database_url,
            redis_url=redis_url,
            storage_root=storage_root,
            transcript_dataset_root=transcript_dataset_root,
            groq_api_keys=groq_api_keys,
            mistral_api_keys=mistral_api_keys,
            groq_model=os.getenv("GROQ_MODEL", "whisper-large-v3"),
            mistral_model=os.getenv("MISTRAL_MODEL", "voxtral-mini-latest"),
            whisper_cpp_base_url=whisper_cpp_base_url,
            whisper_cpp_model_path=os.getenv("WHISPER_CPP_MODEL_PATH") or None,
            whisper_cpp_temperature=_parse_float("WHISPER_CPP_TEMPERATURE", 0.0),
            whisper_cpp_temperature_inc=_parse_float("WHISPER_CPP_TEMPERATURE_INC", 0.2),
            request_timeout_sec=_parse_int("REQUEST_TIMEOUT_SEC", 300),
            provider_max_retries=_parse_int("PROVIDER_MAX_RETRIES", 3),
            chunk_duration_sec=chunk_duration,
            chunk_overlap_sec=chunk_overlap,
            max_file_size_mb=max_file_size,
            max_parallel_chunks=_parse_int("MAX_PARALLEL_CHUNKS", 3),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            job_retention_days=_parse_int("JOB_RETENTION_DAYS", 7),
            queue_name=os.getenv("QUEUE_NAME", "audio-transcript:jobs"),
            dynamic_whisper_cpp_load=os.getenv("DYNAMIC_WHISPER_CPP_LOAD", "false").lower() == "true",
        )

    def validate(self) -> None:
        """Run additional validation after construction.
        
        Call this if Settings was created directly (not via from_env).
        """
        _validate_postgres_url(self.database_url, "database_url")
        _validate_redis_url(self.redis_url, "redis_url")
        _validate_http_url(self.whisper_cpp_base_url, "whisper_cpp_base_url")
        
        if self.chunk_overlap_sec >= self.chunk_duration_sec:
            raise ConfigurationError("chunk_overlap_sec must be less than chunk_duration_sec")
        if self.max_file_size_mb <= 0:
            raise ConfigurationError("max_file_size_mb must be positive")
```

**Test:**
```python
import pytest
from audio_transcript.config import Settings
from audio_transcript.domain.errors import ConfigurationError


def test_settings_rejects_invalid_database_url(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "not-a-url")
    monkeypatch.setenv("GROQ_API_KEYS", "key1")
    
    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env()
    
    assert "DATABASE_URL" in str(exc_info.value)
    assert "scheme" in str(exc_info.value).lower()


def test_settings_rejects_invalid_redis_url(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    monkeypatch.setenv("REDIS_URL", "http://localhost:6379")  # Wrong scheme
    monkeypatch.setenv("GROQ_API_KEYS", "key1")
    
    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env()
    
    assert "REDIS_URL" in str(exc_info.value)
    assert "http" in str(exc_info.value)


def test_settings_rejects_overlap_greater_than_duration(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    monkeypatch.setenv("GROQ_API_KEYS", "key1")
    monkeypatch.setenv("CHUNK_DURATION_SEC", "60")
    monkeypatch.setenv("CHUNK_OVERLAP_SEC", "120")
    
    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env()
    
    assert "CHUNK_OVERLAP_SEC" in str(exc_info.value)
```

---

## 5.3 Add Default QUEUE_NAME

**File:** `src/audio_transcript/config.py`

**Problem:**
The queue name already has a default in the dataclass:
```python
queue_name: str = "audio-transcript:jobs"
```

But there's no documentation about what this controls or how to change it. The default is also used in `from_env()`:
```python
queue_name=os.getenv("QUEUE_NAME", "audio-transcript:jobs"),
```

**Solution:**
This is already implemented correctly. Just add documentation.

**Implementation:**

**Add to README.md configuration section:**
```markdown
## Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SERVICE_API_KEY` | API key for authenticating requests | `sk-abc123...` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost/db` |
| `GROQ_API_KEYS` | Comma-separated Groq API keys | `gsk_key1,gsk_key2` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis connection for job queue |
| `QUEUE_NAME` | `audio-transcript:jobs` | Redis key prefix for job queue |
| `STORAGE_ROOT` | `./data` | Directory for uploaded files |
| `TRANSCRIPT_DATASET_ROOT` | `./transcript_dataset` | Directory for Parquet output |
| `MAX_FILE_SIZE_MB` | `25` | Maximum upload size in MB |
| `CHUNK_DURATION_SEC` | `600` | Chunk size for large files (seconds) |
| `CHUNK_OVERLAP_SEC` | `5` | Overlap between chunks (seconds) |
| `MAX_PARALLEL_CHUNKS` | `3` | Concurrent chunk transcriptions |
| `REQUEST_TIMEOUT_SEC` | `300` | Provider API timeout |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
```

**Add .env.example file:**
```bash
# Required
SERVICE_API_KEY=your-secret-api-key
DATABASE_URL=postgresql://user:password@localhost:5432/audio_transcript
GROQ_API_KEYS=gsk_key1,gsk_key2

# Optional - uncomment to override defaults
# MISTRAL_API_KEYS=mk_key1
# REDIS_URL=redis://localhost:6379/0
# QUEUE_NAME=audio-transcript:jobs
# STORAGE_ROOT=./data
# TRANSCRIPT_DATASET_ROOT=./transcript_dataset
# MAX_FILE_SIZE_MB=25
# CHUNK_DURATION_SEC=600
# CHUNK_OVERLAP_SEC=5
# MAX_PARALLEL_CHUNKS=3
# REQUEST_TIMEOUT_SEC=300
# LOG_LEVEL=INFO

# whisper.cpp fallback (optional)
# WHISPER_CPP_BASE_URL=http://127.0.0.1:8334
# WHISPER_CPP_MODEL_PATH=/models/ggml-large-v3.bin
```

---

## Verification Checklist

After implementing Phase 5:

- [ ] `pip install -e ".[dev]"` works from clean environment
- [ ] `requirements.txt` no longer exists
- [ ] Invalid DATABASE_URL fails with clear message at startup
- [ ] Invalid REDIS_URL fails with clear message at startup
- [ ] Invalid WHISPER_CPP_BASE_URL fails with clear message
- [ ] CHUNK_OVERLAP >= CHUNK_DURATION fails at startup
- [ ] `.env.example` documents all settings
- [ ] README configuration section is complete

---

## Files Modified

| File | Change |
|------|--------|
| `requirements.txt` | DELETE |
| `pyproject.toml` | Complete dependency spec, add extras |
| `src/audio_transcript/config.py` | Add URL validation, improve error messages |
| `.env.example` | CREATE - document all settings |
| `README.md` | Add configuration reference table |

---

## Migration Notes

For existing deployments:

1. If using `pip install -r requirements.txt`:
   ```bash
   # Old way (deprecated)
   pip install -r requirements.txt
   
   # New way
   pip install ".[prod]"
   ```

2. If using Docker:
   ```dockerfile
   # Old
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   # New
   COPY pyproject.toml .
   COPY src/ src/
   RUN pip install ".[prod]"
   ```

3. For CI/CD pinned builds:
   ```bash
   pip install pip-tools
   pip-compile pyproject.toml --extra=prod -o requirements.lock
   pip install -r requirements.lock
   ```
