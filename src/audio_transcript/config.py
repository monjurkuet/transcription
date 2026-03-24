"""Configuration loading."""

from __future__ import annotations

import os
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
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _parse_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc


def _validate_url(value: str, name: str, allowed_schemes: List[str]) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in allowed_schemes:
        raise ConfigurationError(f"{name} must use {'/'.join(allowed_schemes)} scheme")
    if parsed.scheme in {"postgres", "postgresql", "redis", "rediss", "http", "https"} and not parsed.netloc:
        raise ConfigurationError(f"{name} must include a host")
    return value


@dataclass
class Settings:
    """Runtime settings loaded from environment variables.

    The settings object centralizes API auth, infrastructure endpoints,
    provider configuration, processing thresholds, and logging behavior so
    the API and worker share the same runtime contract.
    """

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
    log_level: str = "INFO"
    log_format: str = "text"
    job_retention_days: int = 7
    queue_name: str = "audio-transcript:jobs"
    dynamic_whisper_cpp_load: bool = False
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    db_pool_timeout_sec: int = 30
    max_parallel_chunks: int = 3

    def validate(self) -> None:
        _validate_url(self.database_url, "DATABASE_URL", ["postgres", "postgresql"])
        _validate_url(self.redis_url, "REDIS_URL", ["redis", "rediss"])
        _validate_url(self.whisper_cpp_base_url, "WHISPER_CPP_BASE_URL", ["http", "https"])
        if self.chunk_duration_sec <= 0:
            raise ConfigurationError("CHUNK_DURATION_SEC must be > 0")
        if self.chunk_overlap_sec >= self.chunk_duration_sec:
            raise ConfigurationError("CHUNK_OVERLAP_SEC must be less than CHUNK_DURATION_SEC")
        if self.max_file_size_mb <= 0:
            raise ConfigurationError("MAX_FILE_SIZE_MB must be > 0")
        if self.max_parallel_chunks <= 0:
            raise ConfigurationError("MAX_PARALLEL_CHUNKS must be > 0")
        if self.request_timeout_sec <= 0:
            raise ConfigurationError("REQUEST_TIMEOUT_SEC must be > 0")
        if self.provider_max_retries <= 0:
            raise ConfigurationError("PROVIDER_MAX_RETRIES must be > 0")
        if self.log_format not in {"text", "json"}:
            raise ConfigurationError("LOG_FORMAT must be either 'text' or 'json'")
        if self.db_pool_min_size <= 0:
            raise ConfigurationError("DB_POOL_MIN_SIZE must be > 0")
        if self.db_pool_max_size < self.db_pool_min_size:
            raise ConfigurationError("DB_POOL_MAX_SIZE must be >= DB_POOL_MIN_SIZE")
        if self.db_pool_timeout_sec <= 0:
            raise ConfigurationError("DB_POOL_TIMEOUT_SEC must be > 0")

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from env and validate them."""
        load_dotenv()
        service_api_key = os.getenv("SERVICE_API_KEY", "").strip()
        if not service_api_key:
            raise ConfigurationError("SERVICE_API_KEY is required")
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise ConfigurationError("DATABASE_URL is required")

        storage_root = Path(os.getenv("STORAGE_ROOT", "./data")).resolve()
        groq_api_keys = _parse_csv_env("GROQ_API_KEYS")
        if not groq_api_keys:
            single_groq = os.getenv("GROQ_API_KEY", "").strip()
            if single_groq:
                groq_api_keys = [single_groq]

        mistral_api_keys = _parse_csv_env("MISTRAL_API_KEYS")
        if not groq_api_keys and not mistral_api_keys:
            raise ConfigurationError("At least one remote provider key is required")

        settings = cls(
            service_api_key=service_api_key,
            database_url=database_url,
            redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            storage_root=storage_root,
            transcript_dataset_root=Path(os.getenv("TRANSCRIPT_DATASET_ROOT", "./transcript_dataset")).resolve(),
            groq_api_keys=groq_api_keys,
            mistral_api_keys=mistral_api_keys,
            groq_model=os.getenv("GROQ_MODEL", "whisper-large-v3"),
            mistral_model=os.getenv("MISTRAL_MODEL", "voxtral-mini-latest"),
            whisper_cpp_base_url=os.getenv("WHISPER_CPP_BASE_URL", "http://127.0.0.1:8334").rstrip("/"),
            whisper_cpp_model_path=os.getenv("WHISPER_CPP_MODEL_PATH") or None,
            whisper_cpp_temperature=_parse_float("WHISPER_CPP_TEMPERATURE", 0.0),
            whisper_cpp_temperature_inc=_parse_float("WHISPER_CPP_TEMPERATURE_INC", 0.2),
            request_timeout_sec=_parse_int("REQUEST_TIMEOUT_SEC", 300),
            provider_max_retries=_parse_int("PROVIDER_MAX_RETRIES", 3),
            chunk_duration_sec=_parse_int("CHUNK_DURATION_SEC", 600),
            chunk_overlap_sec=_parse_int("CHUNK_OVERLAP_SEC", 5),
            max_file_size_mb=_parse_int("MAX_FILE_SIZE_MB", 25),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "text").strip().lower() or "text",
            job_retention_days=_parse_int("JOB_RETENTION_DAYS", 7),
            queue_name=os.getenv("QUEUE_NAME", "audio-transcript:jobs"),
            dynamic_whisper_cpp_load=os.getenv("DYNAMIC_WHISPER_CPP_LOAD", "false").lower() == "true",
            db_pool_min_size=_parse_int("DB_POOL_MIN_SIZE", 2),
            db_pool_max_size=_parse_int("DB_POOL_MAX_SIZE", 10),
            db_pool_timeout_sec=_parse_int("DB_POOL_TIMEOUT_SEC", 30),
            max_parallel_chunks=_parse_int("MAX_PARALLEL_CHUNKS", 3),
        )
        settings.validate()
        return settings
