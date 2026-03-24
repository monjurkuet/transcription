import pytest

from audio_transcript.domain.errors import ConfigurationError
from audio_transcript.config import Settings


def test_settings_load_database_pool_env(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1")
    monkeypatch.setenv("DB_POOL_MIN_SIZE", "3")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "12")
    monkeypatch.setenv("DB_POOL_TIMEOUT_SEC", "45")
    monkeypatch.setenv("MAX_PARALLEL_CHUNKS", "6")

    settings = Settings.from_env()

    assert settings.db_pool_min_size == 3
    assert settings.db_pool_max_size == 12
    assert settings.db_pool_timeout_sec == 45
    assert settings.max_parallel_chunks == 6


def test_settings_rejects_invalid_database_url(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "not-a-url")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1")

    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env()

    assert "DATABASE_URL" in str(exc_info.value)


def test_settings_rejects_invalid_redis_url(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    monkeypatch.setenv("REDIS_URL", "http://localhost:6379")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1")

    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env()

    assert "REDIS_URL" in str(exc_info.value)


def test_settings_rejects_invalid_whisper_cpp_url(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    monkeypatch.setenv("WHISPER_CPP_BASE_URL", "redis://localhost:8334")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1")

    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env()

    assert "WHISPER_CPP_BASE_URL" in str(exc_info.value)


def test_settings_rejects_non_integer_values(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1")
    monkeypatch.setenv("MAX_PARALLEL_CHUNKS", "abc")

    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env()

    assert "MAX_PARALLEL_CHUNKS" in str(exc_info.value)


def test_settings_rejects_overlap_greater_than_duration(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1")
    monkeypatch.setenv("CHUNK_DURATION_SEC", "60")
    monkeypatch.setenv("CHUNK_OVERLAP_SEC", "120")

    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env()

    assert "CHUNK_OVERLAP_SEC" in str(exc_info.value)


def test_settings_rejects_non_positive_numeric_values(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "0")

    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env()

    assert "MAX_FILE_SIZE_MB" in str(exc_info.value)
