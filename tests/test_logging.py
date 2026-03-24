import io
import json

from audio_transcript.config import Settings
from audio_transcript.domain.errors import NonRetryableProviderError, RetryableProviderError
from audio_transcript.domain.models import JobStatus, TranscriptResult, TranscriptSegment
from audio_transcript.infra.queue import InMemoryQueueBackend
from audio_transcript.infra.repository import InMemoryJobRepository
from audio_transcript.infra.runtime_state import InMemoryRuntimeState
from audio_transcript.infra.storage import TranscriptArtifactStore
from audio_transcript.logging_utils import (
    clear_context,
    configure_logging,
    set_job_context,
    set_request_context,
)
from audio_transcript.services.router import ProviderRouter
from audio_transcript.services.transcription import DirectoryScanService, RuntimeDependencies, TranscriptionService
from audio_transcript.worker.runner import run_single_iteration
from conftest import FakeChunker, FakeInspector, FakeProvider


def _runtime_for_provider(tmp_path, provider, *, provider_max_retries=3):
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
        provider_max_retries=provider_max_retries,
        log_format="json",
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
    return {
        "settings": settings,
        "repository": repository,
        "queue": queue,
        "service": service,
        "artifact_store": store,
    }


def _create_job(runtime, job_id="job-1", filename="sample.wav"):
    path = runtime["artifact_store"].root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFFfake")
    job = {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
    }
    from audio_transcript.domain.models import JobPayload, TranscriptionJob

    job_model = TranscriptionJob(
        job_id=job["job_id"],
        status=job["status"],
        payload=JobPayload(filename=filename, content_type="audio/wav", source_path=str(path)),
    )
    runtime["repository"].create(job_model)
    runtime["queue"].enqueue(job_id)
    return job_model


def _log_lines(buffer):
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def test_configure_logging_json_includes_context_and_extras():
    stream = io.StringIO()
    configure_logging(log_format="json", stream=stream)
    set_request_context("req-1")
    set_job_context("job-1")

    import logging

    logging.getLogger("audio_transcript.test").info("hello", extra={"provider": "groq", "duration_ms": 12})
    clear_context()

    payload = _log_lines(stream)[0]
    assert payload["request_id"] == "req-1"
    assert payload["job_id"] == "job-1"
    assert payload["provider"] == "groq"
    assert payload["duration_ms"] == 12


def test_configure_logging_text_keeps_human_readable_format():
    stream = io.StringIO()
    configure_logging(log_format="text", stream=stream)
    set_request_context("req-human")
    set_job_context("job-human")

    import logging

    logging.getLogger("audio_transcript.test").info("hello", extra={"provider": "groq"})
    clear_context()

    line = stream.getvalue().strip()
    assert "audio_transcript.test: hello" in line
    assert "req=req-huma" in line
    assert "job=job-huma" in line
    assert "provider=groq" in line


def test_service_success_logs_include_job_id(tmp_path):
    stream = io.StringIO()
    configure_logging(log_format="json", stream=stream)
    provider = FakeProvider(
        "groq",
        outcomes=[
            TranscriptResult(
                text="groq text",
                segments=[TranscriptSegment(start=0.0, end=1.0, text="groq text")],
                provider="groq",
                model="whisper-large-v3",
            )
        ],
    )
    runtime = _runtime_for_provider(tmp_path, provider)
    job = _create_job(runtime, job_id="job-success")

    runtime["service"].process_job(job.job_id)

    events = _log_lines(stream)
    assert any(item["event"] == "provider_attempt_succeeded" and item["job_id"] == "job-success" for item in events)
    assert any(item["event"] == "job_succeeded" and item["job_id"] == "job-success" for item in events)


def test_worker_retryable_logs_include_job_id(tmp_path):
    stream = io.StringIO()
    configure_logging(log_format="json", stream=stream)
    provider = FakeProvider("groq", outcomes=[RetryableProviderError("temporary")])
    runtime = _runtime_for_provider(tmp_path, provider, provider_max_retries=3)
    _create_job(runtime, job_id="job-retry")

    run_single_iteration(runtime, sleep_fn=lambda _: None)

    events = _log_lines(stream)
    assert any(item["event"] == "job_retryable_failure" and item["job_id"] == "job-retry" for item in events)
    assert any(item["event"] == "worker_job_retry_scheduled" and item["job_id"] == "job-retry" for item in events)


def test_worker_terminal_logs_include_job_id(tmp_path):
    stream = io.StringIO()
    configure_logging(log_format="json", stream=stream)
    provider = FakeProvider("groq", outcomes=[NonRetryableProviderError("bad audio")])
    runtime = _runtime_for_provider(tmp_path, provider)
    _create_job(runtime, job_id="job-failed")

    run_single_iteration(runtime, sleep_fn=lambda _: None)

    events = _log_lines(stream)
    assert any(item["event"] == "job_failed" and item["job_id"] == "job-failed" for item in events)
    assert any(item["event"] == "worker_job_failed" and item["job_id"] == "job-failed" for item in events)


def test_directory_scan_logs_summary_counts(tmp_path):
    stream = io.StringIO()
    configure_logging(log_format="json", stream=stream)
    repository = InMemoryJobRepository()
    queue = InMemoryQueueBackend()
    service = DirectoryScanService(repository, queue)
    root = tmp_path / "scan"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "a.wav").write_bytes(b"RIFFfake")
    (nested / "ignore.txt").write_text("ignore")

    service.scan_directory(str(root))

    events = _log_lines(stream)
    summary = next(item for item in events if item["event"] == "scan_completed")
    assert summary["directory_path"] == str(root.resolve())
    assert summary["discovered"] == 2
    assert summary["queued"] == 1
    assert summary["unsupported"] == 1
