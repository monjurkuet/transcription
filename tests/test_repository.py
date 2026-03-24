from contextlib import contextmanager
from datetime import datetime, timezone

from audio_transcript.domain.models import JobPayload, JobStatus, ProviderAttempt, TranscriptionJob

from audio_transcript.infra.repository import InMemoryJobRepository, PostgresJobRepository


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


def test_inmemory_repository_finds_latest_by_exact_source_path():
    repository = InMemoryJobRepository()
    older = TranscriptionJob(
        job_id="job-older",
        status=JobStatus.FAILED,
        payload=JobPayload(filename="sample.wav", content_type="audio/wav", source_path="/tmp/sample.wav"),
    )
    newer = TranscriptionJob(
        job_id="job-newer",
        status=JobStatus.SUCCEEDED,
        payload=JobPayload(filename="sample.wav", content_type="audio/wav", source_path="/tmp/sample.wav"),
    )
    repository.create(older)
    repository.create(newer)

    found = repository.find_latest_by_source_path("/tmp/sample.wav")

    assert found is not None
    assert found.job_id == "job-newer"


def test_postgres_repository_find_latest_by_source_path_queries_most_recent_job():
    repository = PostgresJobRepository.__new__(PostgresJobRepository)

    class FindCursor(FakeCursor):
        def __init__(self):
            super().__init__()
            self.fetchone_calls = 0

        def fetchone(self):
            self.fetchone_calls += 1
            if self.fetchone_calls == 1:
                return {
                    "job_id": "job-1",
                    "status": "succeeded",
                    "source_filename": "sample.wav",
                    "content_type": "audio/wav",
                    "source_path": "/tmp/sample.wav",
                    "model_override": None,
                    "chunk_duration_sec": None,
                    "chunk_overlap_sec": None,
                    "provider": "groq",
                    "model": "whisper-large-v3",
                    "created_at": datetime.now(timezone.utc),
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                    "artifact_uri": None,
                    "artifact_format": None,
                    "summary_text": None,
                    "segment_count": None,
                    "duration_sec": None,
                    "size_bytes": None,
                    "codec": None,
                    "sample_rate": None,
                    "channels": None,
                    "bit_rate": None,
                    "format": None,
                    "file_metadata": None,
                }
            return None

        def fetchall(self):
            return []

    cursor = FindCursor()

    @contextmanager
    def fake_connection():
        yield type("Conn", (), {"cursor": lambda self: cursor})()

    repository._connection = fake_connection

    found = repository.find_latest_by_source_path("/tmp/sample.wav")

    assert found is not None
    assert found.job_id == "job-1"
    assert "WHERE source_path = %s" in cursor.calls[0][0]
