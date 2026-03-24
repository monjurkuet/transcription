from audio_transcript.config import Settings


def test_settings_load_database_pool_env(monkeypatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1")
    monkeypatch.setenv("DB_POOL_MIN_SIZE", "3")
    monkeypatch.setenv("DB_POOL_MAX_SIZE", "12")
    monkeypatch.setenv("DB_POOL_TIMEOUT_SEC", "45")

    settings = Settings.from_env()

    assert settings.db_pool_min_size == 3
    assert settings.db_pool_max_size == 12
    assert settings.db_pool_timeout_sec == 45
