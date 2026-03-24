import io

from audio_transcript.api.app import create_app
from audio_transcript.config import Settings
from audio_transcript.domain.errors import AudioProcessingError, NonRetryableProviderError, RetryableProviderError, StorageError
from audio_transcript.domain.models import JobStatus, TranscriptResult, TranscriptSegment
from audio_transcript.infra.queue import InMemoryQueueBackend
from audio_transcript.infra.repository import InMemoryJobRepository
from audio_transcript.infra.runtime_state import InMemoryRuntimeState
from audio_transcript.infra.storage import TranscriptArtifactStore
from audio_transcript.services.router import ProviderRouter
from audio_transcript.services.transcription import RuntimeDependencies, TranscriptionService
from audio_transcript.worker.runner import calculate_backoff, run_single_iteration
from conftest import FakeChunker, FakeInspector, FakeProvider, build_test_app


def test_api_job_lifecycle(tmp_path):
    app, runtime = build_test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "secret"},
        data={"file": (io.BytesIO(b"RIFFfake"), "sample.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    job_id = response.get_json()["job"]["id"]

    status_response = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "secret"})
    assert status_response.status_code == 200
    assert status_response.get_json()["job"]["status"] == "queued"

    run_single_iteration(runtime)

    status_response = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "secret"})
    assert status_response.get_json()["job"]["status"] == "succeeded"

    result_response = client.get(f"/v1/jobs/{job_id}/result", headers={"X-API-Key": "secret"})
    assert result_response.status_code == 200
    assert result_response.get_json()["transcript"]["provider"] == "groq"
    assert (tmp_path / "dataset").exists()


def test_api_rejects_invalid_auth(tmp_path):
    app, _ = build_test_app(tmp_path)
    client = app.test_client()
    response = client.get("/v1/providers/status", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_api_lists_jobs_with_search(tmp_path):
    app, runtime = build_test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "secret"},
        data={"file": (io.BytesIO(b"RIFFfake"), "sample.wav")},
        content_type="multipart/form-data",
    )
    job_id = response.get_json()["job"]["id"]
    run_single_iteration(runtime)

    response = client.get("/v1/jobs?provider=groq&search=groq", headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    jobs = response.get_json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["id"] == job_id


def test_api_internal_errors_do_not_leak_details(tmp_path):
    app, _ = build_test_app(tmp_path)

    @app.get("/boom")
    def boom():
        raise RuntimeError("/home/test Traceback secret")

    client = app.test_client()
    response = client.get("/boom")

    assert response.status_code == 500
    payload = response.get_json()["error"]
    assert payload["code"] == "internal_error"
    assert payload["message"] == "An unexpected error occurred"
    assert "/home/test" not in payload["message"]
    assert "Traceback" not in payload["message"]


def test_api_audio_processing_error_returns_422(tmp_path):
    app, _ = build_test_app(tmp_path)

    @app.get("/audio-boom")
    def audio_boom():
        raise AudioProcessingError("Failed to read audio metadata from 'bad.wav': invalid data")

    client = app.test_client()
    response = client.get("/audio-boom")

    assert response.status_code == 422
    payload = response.get_json()["error"]
    assert payload["code"] == "audio_processing_error"
    assert "bad.wav" in payload["message"]


def test_api_storage_error_returns_generic_message(tmp_path):
    app, _ = build_test_app(tmp_path)

    @app.get("/storage-boom")
    def storage_boom():
        raise StorageError("disk full at /secret/path")

    client = app.test_client()
    response = client.get("/storage-boom")

    assert response.status_code == 500
    payload = response.get_json()["error"]
    assert payload["code"] == "storage_error"
    assert payload["message"] == "Failed to persist transcript artifacts"


def test_api_result_supports_segment_pagination(tmp_path):
    app, runtime = build_test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "secret"},
        data={"file": (io.BytesIO(b"RIFFfake"), "sample.wav")},
        content_type="multipart/form-data",
    )
    job_id = response.get_json()["job"]["id"]
    run_single_iteration(runtime)

    result_response = client.get(
        f"/v1/jobs/{job_id}/result?segment_offset=0&segment_limit=1",
        headers={"X-API-Key": "secret"},
    )
    payload = result_response.get_json()

    assert result_response.status_code == 200
    assert len(payload["transcript"]["segments"]) == 1
    assert payload["pagination"] == {"offset": 0, "limit": 1, "total": 1, "has_more": False}


def test_api_result_rejects_invalid_pagination(tmp_path):
    app, runtime = build_test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "secret"},
        data={"file": (io.BytesIO(b"RIFFfake"), "sample.wav")},
        content_type="multipart/form-data",
    )
    job_id = response.get_json()["job"]["id"]
    run_single_iteration(runtime)

    bad_offset = client.get(
        f"/v1/jobs/{job_id}/result?segment_offset=-1",
        headers={"X-API-Key": "secret"},
    )
    bad_limit = client.get(
        f"/v1/jobs/{job_id}/result?segment_limit=0",
        headers={"X-API-Key": "secret"},
    )

    assert bad_offset.status_code == 400
    assert bad_offset.get_json()["error"]["message"] == "segment_offset must be >= 0"
    assert bad_limit.status_code == 400
    assert bad_limit.get_json()["error"]["message"] == "segment_limit must be > 0"


def test_worker_retries_retryable_job_until_success(tmp_path):
    repository = InMemoryJobRepository()
    queue = InMemoryQueueBackend()
    runtime_state = InMemoryRuntimeState()
    store = TranscriptArtifactStore(tmp_path / "artifacts", tmp_path / "dataset")
    settings = Settings(
        service_api_key="secret",
        database_url="postgresql://unused",
        redis_url="redis://unused",
        storage_root=tmp_path / "artifacts",
        transcript_dataset_root=tmp_path / "dataset",
        groq_api_keys=["gsk_1"],
        mistral_api_keys=[],
        provider_max_retries=3,
    )
    provider = FakeProvider(
        "groq",
        outcomes=[
            RetryableProviderError("temporary 1"),
            RetryableProviderError("temporary 2"),
            TranscriptResult(
                text="groq text",
                segments=[TranscriptSegment(start=0.0, end=1.0, text="groq text")],
                provider="groq",
            ),
        ],
    )
    service = TranscriptionService(
        RuntimeDependencies(
            settings=settings,
            repository=repository,
            artifact_store=store,
            runtime_state=runtime_state,
            router=ProviderRouter(["groq"]),
            remote_providers={"groq": provider},
            fallback_provider=None,
            inspector=FakeInspector(),
            chunker=FakeChunker(),
        )
    )
    runtime = {
        "settings": settings,
        "repository": repository,
        "queue": queue,
        "service": service,
        "artifact_store": store,
    }
    response = create_app(
        settings,
        repository=repository,
        queue=queue,
        artifact_store=store,
        runtime_state=runtime_state,
        service=service,
        providers={"groq": provider},
        fallback_provider=None,
    ).test_client().post(
        "/v1/jobs",
        headers={"X-API-Key": "secret"},
        data={"file": (io.BytesIO(b"RIFFfake"), "sample.wav")},
        content_type="multipart/form-data",
    )
    job_id = response.get_json()["job"]["id"]

    sleeps = []
    assert run_single_iteration(runtime, sleep_fn=sleeps.append) is True
    assert queue.get_retry_count(job_id) == 1
    assert sleeps == [2]
    assert repository.get(job_id).status == JobStatus.QUEUED

    assert run_single_iteration(runtime, sleep_fn=sleeps.append) is True
    assert queue.get_retry_count(job_id) == 2
    assert sleeps == [2, 4]

    assert run_single_iteration(runtime, sleep_fn=sleeps.append) is True
    job = repository.get(job_id)
    assert job.status == JobStatus.SUCCEEDED
    assert queue.get_retry_count(job_id) == 0
    assert queue.get_dlq_jobs() == []


def test_worker_moves_non_retryable_job_to_dlq(tmp_path):
    repository = InMemoryJobRepository()
    queue = InMemoryQueueBackend()
    runtime_state = InMemoryRuntimeState()
    store = TranscriptArtifactStore(tmp_path / "artifacts", tmp_path / "dataset")
    settings = Settings(
        service_api_key="secret",
        database_url="postgresql://unused",
        redis_url="redis://unused",
        storage_root=tmp_path / "artifacts",
        transcript_dataset_root=tmp_path / "dataset",
        groq_api_keys=["gsk_1"],
        mistral_api_keys=[],
    )
    provider = FakeProvider("groq", outcomes=[NonRetryableProviderError("bad audio")])
    service = TranscriptionService(
        RuntimeDependencies(
            settings=settings,
            repository=repository,
            artifact_store=store,
            runtime_state=runtime_state,
            router=ProviderRouter(["groq"]),
            remote_providers={"groq": provider},
            fallback_provider=None,
            inspector=FakeInspector(),
            chunker=FakeChunker(),
        )
    )
    client = create_app(
        settings,
        repository=repository,
        queue=queue,
        artifact_store=store,
        runtime_state=runtime_state,
        service=service,
        providers={"groq": provider},
        fallback_provider=None,
    ).test_client()
    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "secret"},
        data={"file": (io.BytesIO(b"RIFFfake"), "sample.wav")},
        content_type="multipart/form-data",
    )
    job_id = response.get_json()["job"]["id"]

    runtime = {
        "settings": settings,
        "repository": repository,
        "queue": queue,
        "service": service,
        "artifact_store": store,
    }
    assert run_single_iteration(runtime, sleep_fn=lambda _: None) is True
    job = repository.get(job_id)
    assert job.status == JobStatus.FAILED
    assert queue.get_retry_count(job_id) == 0
    dlq = queue.get_dlq_jobs()
    assert len(dlq) == 1
    assert dlq[0]["job_id"] == job_id
    assert dlq[0]["error"] == "bad audio"


def test_worker_moves_retry_exhausted_job_to_dlq(tmp_path):
    repository = InMemoryJobRepository()
    queue = InMemoryQueueBackend()
    runtime_state = InMemoryRuntimeState()
    store = TranscriptArtifactStore(tmp_path / "artifacts", tmp_path / "dataset")
    settings = Settings(
        service_api_key="secret",
        database_url="postgresql://unused",
        redis_url="redis://unused",
        storage_root=tmp_path / "artifacts",
        transcript_dataset_root=tmp_path / "dataset",
        groq_api_keys=["gsk_1"],
        mistral_api_keys=[],
        provider_max_retries=3,
    )
    provider = FakeProvider(
        "groq",
        outcomes=[
            RetryableProviderError("temporary 1"),
            RetryableProviderError("temporary 2"),
            RetryableProviderError("temporary 3"),
        ],
    )
    service = TranscriptionService(
        RuntimeDependencies(
            settings=settings,
            repository=repository,
            artifact_store=store,
            runtime_state=runtime_state,
            router=ProviderRouter(["groq"]),
            remote_providers={"groq": provider},
            fallback_provider=None,
            inspector=FakeInspector(),
            chunker=FakeChunker(),
        )
    )
    client = create_app(
        settings,
        repository=repository,
        queue=queue,
        artifact_store=store,
        runtime_state=runtime_state,
        service=service,
        providers={"groq": provider},
        fallback_provider=None,
    ).test_client()
    response = client.post(
        "/v1/jobs",
        headers={"X-API-Key": "secret"},
        data={"file": (io.BytesIO(b"RIFFfake"), "sample.wav")},
        content_type="multipart/form-data",
    )
    job_id = response.get_json()["job"]["id"]
    runtime = {
        "settings": settings,
        "repository": repository,
        "queue": queue,
        "service": service,
        "artifact_store": store,
    }

    sleeps = []
    assert run_single_iteration(runtime, sleep_fn=sleeps.append) is True
    assert run_single_iteration(runtime, sleep_fn=sleeps.append) is True
    assert run_single_iteration(runtime, sleep_fn=sleeps.append) is True

    job = repository.get(job_id)
    assert job.status == JobStatus.FAILED
    assert job.error == "Retry limit exceeded"
    assert queue.get_retry_count(job_id) == 0
    assert sleeps == [2, 4]
    dlq = queue.get_dlq_jobs()
    assert len(dlq) == 1
    assert dlq[0]["job_id"] == job_id
    assert dlq[0]["retry_count"] == 3


def test_calculate_backoff_grows_exponentially():
    assert calculate_backoff(1) == 2
    assert calculate_backoff(2) == 4
    assert calculate_backoff(3) == 8
