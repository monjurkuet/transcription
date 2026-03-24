from contextlib import contextmanager
from datetime import datetime, timezone

from audio_transcript.domain.models import JobPayload, JobStatus, ProviderAttempt, TranscriptionJob

from audio_transcript.infra.repository import PostgresJobRepository


class FakeCursor:
    def __init__(self):
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((query, params))
        self.query = query
        self.params = params

    def fetchone(self):
        return {"ok": 1}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def cursor(self):
        return FakeCursor()


class FakePool:
    def __init__(self):
        self.stats = {"pool_size": 4, "pool_available": 3}

    def get_stats(self):
        return self.stats


def test_postgres_repository_healthcheck_includes_pool_stats():
    repository = PostgresJobRepository.__new__(PostgresJobRepository)
    repository._pool = FakePool()

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    repository._connection = fake_connection

    status = repository.healthcheck()

    assert status == {
        "postgres": "ok",
        "postgres_pool_size": "4",
        "postgres_pool_available": "3",
    }


def test_repository_bootstraps_attempt_constraint_with_idempotent_block():
    repository = PostgresJobRepository.__new__(PostgresJobRepository)
    cursor = FakeCursor()

    repository._ensure_attempt_constraints(cursor)

    assert "DO $$" in cursor.query
    assert "uq_job_attempts_job_provider_started" in cursor.query


def test_repository_upserts_attempts_instead_of_deleting_all_rows():
    repository = PostgresJobRepository.__new__(PostgresJobRepository)
    cursor = FakeCursor()
    job = TranscriptionJob(
        job_id="job-1",
        status=JobStatus.QUEUED,
        payload=JobPayload(filename="sample.wav", content_type="audio/wav", source_path="/tmp/sample.wav"),
        attempts=[
            ProviderAttempt(
                provider="groq",
                key_id="gsk***",
                started_at=datetime.now(timezone.utc),
                success=True,
                retryable=False,
            )
        ],
    )

    repository._upsert_attempts(cursor, job)

    assert len(cursor.calls) == 1
    query, params = cursor.calls[0]
    assert "ON CONFLICT (job_id, provider, started_at) DO UPDATE" in query
    assert "DELETE FROM job_attempts" not in query
    assert params[0] == "job-1"
