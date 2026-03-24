from datetime import timezone

import pytest

from audio_transcript.domain.models import FileMetadata, JobPayload, JobStatus, ProviderAttempt, TranscriptionJob
from audio_transcript.infra.repository import PostgresJobRepository

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def postgres_container():
    testcontainers = pytest.importorskip("testcontainers.postgres")
    try:
        with testcontainers.PostgresContainer("postgres:15-alpine") as postgres:
            yield postgres
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Postgres integration tests unavailable: {exc}")


@pytest.fixture
def repository(postgres_container):
    repo = PostgresJobRepository(postgres_container.get_connection_url())
    yield repo
    with repo._connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE jobs, job_attempts CASCADE")
    repo.close()


def test_postgres_repository_round_trip(repository):
    job = TranscriptionJob(
        job_id="repo-1",
        status=JobStatus.QUEUED,
        payload=JobPayload(filename="sample.wav", content_type="audio/wav", source_path="/tmp/sample.wav"),
    )
    repository.create(job)

    loaded = repository.get("repo-1")
    assert loaded.job_id == "repo-1"
    assert loaded.payload.filename == "sample.wav"
    assert loaded.status == JobStatus.QUEUED


def test_postgres_repository_upserts_attempts(repository):
    job = TranscriptionJob(
        job_id="repo-2",
        status=JobStatus.PROCESSING,
        payload=JobPayload(filename="sample.wav", content_type="audio/wav", source_path="/tmp/sample.wav"),
    )
    attempt_started = job.created_at.replace(tzinfo=timezone.utc) if job.created_at.tzinfo is None else job.created_at
    job.attempts.append(
        ProviderAttempt(
            provider="groq",
            started_at=attempt_started,
            success=False,
            retryable=True,
            error="rate limited",
        )
    )
    repository.create(job)

    loaded = repository.get("repo-2")
    loaded.attempts[0].success = True
    loaded.attempts[0].retryable = False
    loaded.attempts[0].error = None
    repository.save(loaded)

    reloaded = repository.get("repo-2")
    assert len(reloaded.attempts) == 1
    assert reloaded.attempts[0].success is True
    assert reloaded.attempts[0].retryable is False


def test_postgres_repository_list_filters_and_healthcheck(repository):
    succeeded = TranscriptionJob(
        job_id="repo-3",
        status=JobStatus.SUCCEEDED,
        payload=JobPayload(filename="ok.wav", content_type="audio/wav", source_path="/tmp/ok.wav"),
        provider="groq",
        summary_text="hello transcript",
        file_metadata=FileMetadata(
            filename="ok.wav",
            path="/tmp/ok.wav",
            size_bytes=1,
            duration=1.0,
            format="wav",
            bit_rate=0,
            codec="pcm",
            sample_rate=16000,
            channels=1,
        ),
    )
    repository.create(succeeded)

    queued = TranscriptionJob(
        job_id="repo-4",
        status=JobStatus.QUEUED,
        payload=JobPayload(filename="wait.wav", content_type="audio/wav", source_path="/tmp/wait.wav"),
    )
    repository.create(queued)

    assert len(repository.list_jobs(status="succeeded")) == 1
    assert len(repository.list_jobs(provider="groq")) == 1
    assert len(repository.list_jobs(search="transcript")) == 1
    assert repository.healthcheck()["postgres"] == "ok"


def test_postgres_repository_finds_latest_by_source_path(repository):
    older = TranscriptionJob(
        job_id="repo-5",
        status=JobStatus.FAILED,
        payload=JobPayload(filename="same.wav", content_type="audio/wav", source_path="/tmp/same.wav"),
    )
    repository.create(older)

    newer = TranscriptionJob(
        job_id="repo-6",
        status=JobStatus.SUCCEEDED,
        payload=JobPayload(filename="same.wav", content_type="audio/wav", source_path="/tmp/same.wav"),
    )
    repository.create(newer)

    found = repository.find_latest_by_source_path("/tmp/same.wav")

    assert found is not None
    assert found.job_id == "repo-6"
    assert found.status == JobStatus.SUCCEEDED
