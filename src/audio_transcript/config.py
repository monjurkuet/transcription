"""Configuration loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from .domain.errors import ConfigurationError


def _parse_csv_env(name: str) -> List[str]:
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _parse_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


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
    log_level: str = "INFO"
    job_retention_days: int = 7
    queue_name: str = "audio-transcript:jobs"
    dynamic_whisper_cpp_load: bool = False

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

        return cls(
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
            job_retention_days=_parse_int("JOB_RETENTION_DAYS", 7),
            queue_name=os.getenv("QUEUE_NAME", "audio-transcript:jobs"),
            dynamic_whisper_cpp_load=os.getenv("DYNAMIC_WHISPER_CPP_LOAD", "false").lower() == "true",
        )
