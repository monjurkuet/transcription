from pathlib import Path

from audio_transcript.config import Settings
from audio_transcript.domain.errors import NonRetryableProviderError, RetryableProviderError
from audio_transcript.domain.models import JobPayload, JobStatus, TranscriptionJob, TranscriptResult, TranscriptSegment
from audio_transcript.infra.repository import InMemoryJobRepository
from audio_transcript.infra.runtime_state import InMemoryRuntimeState
from audio_transcript.infra.storage import TranscriptArtifactStore
from audio_transcript.services.audio import AudioChunker, AudioInspector
from audio_transcript.services.router import ProviderRouter
from audio_transcript.services.transcription import RuntimeDependencies, TranscriptionService


class StaticInspector(AudioInspector):
    def get_file_metadata(self, file_path: Path):
        return super().get_file_metadata(file_path) if False else type(
            "Meta",
            (),
            {
                "to_dict": lambda self: {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "duration": 1.0,
                    "format": file_path.suffix.lstrip("."),
                    "bit_rate": 0,
                    "codec": "pcm",
                    "sample_rate": 16000,
                    "channels": 1,
                }
            },
        )()


class FakeInspector(AudioInspector):
    def get_file_metadata(self, file_path: Path):
        from audio_transcript.domain.models import FileMetadata

        return FileMetadata(
            filename=file_path.name,
            path=str(file_path),
            size_bytes=file_path.stat().st_size,
            duration=1.0,
            format=file_path.suffix.lstrip("."),
            bit_rate=0,
            codec="pcm",
            sample_rate=16000,
            channels=1,
        )


class FakeProvider:
    def __init__(self, name, result=None, error=None):
        self.provider_name = name
        self.result = result
        self.error = error

    def transcribe(self, audio_path, content_type, model_override=None):
        if self.error:
            raise self.error
        return self.result

    def status(self):
        return {"provider": self.provider_name}


def test_service_falls_back_to_whisper_cpp(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")

    repository = InMemoryJobRepository()
    runtime_state = InMemoryRuntimeState()
    store = TranscriptArtifactStore(tmp_path / "artifacts", tmp_path / "dataset")
    settings = Settings(
        service_api_key="secret",
        database_url="postgresql://unused",
        redis_url="redis://unused",
        storage_root=tmp_path / "artifacts",
        transcript_dataset_root=tmp_path / "dataset",
        groq_api_keys=["gsk_1"],
        mistral_api_keys=["ms_1"],
    )
    service = TranscriptionService(
        RuntimeDependencies(
            settings=settings,
            repository=repository,
            artifact_store=store,
            runtime_state=runtime_state,
            router=ProviderRouter(["groq", "mistral"]),
            remote_providers={
                "groq": FakeProvider("groq", error=RetryableProviderError("groq down")),
                "mistral": FakeProvider("mistral", error=RetryableProviderError("mistral down")),
            },
            fallback_provider=FakeProvider(
                "whisper_cpp",
                result=TranscriptResult(
                    text="fallback text",
                    segments=[TranscriptSegment(start=0.0, end=1.0, text="fallback text")],
                    provider="whisper_cpp",
                ),
            ),
            inspector=FakeInspector(),
            chunker=AudioChunker(FakeInspector()),
        )
    )

    job = TranscriptionJob(
        job_id="job-1",
        status=JobStatus.QUEUED,
        payload=JobPayload(
            filename=audio_path.name,
            content_type="audio/wav",
            source_path=str(audio_path),
        ),
    )
    repository.create(job)
    processed = service.process_job(job.job_id)

    assert processed.status == JobStatus.SUCCEEDED
    assert processed.result_path is not None
    assert (tmp_path / "dataset").exists()
    stored = repository.get(job.job_id)
    assert [attempt.provider for attempt in stored.attempts] == ["groq", "mistral", "whisper_cpp"]


def test_retryable_failure_persists_job_as_queued(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")

    repository = InMemoryJobRepository()
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
    service = TranscriptionService(
        RuntimeDependencies(
            settings=settings,
            repository=repository,
            artifact_store=store,
            runtime_state=runtime_state,
            router=ProviderRouter(["groq"]),
            remote_providers={"groq": FakeProvider("groq", error=RetryableProviderError("temporary failure"))},
            fallback_provider=None,
            inspector=FakeInspector(),
            chunker=AudioChunker(FakeInspector()),
        )
    )
    job = TranscriptionJob(
        job_id="job-retry",
        status=JobStatus.QUEUED,
        payload=JobPayload(
            filename=audio_path.name,
            content_type="audio/wav",
            source_path=str(audio_path),
        ),
    )
    repository.create(job)

    try:
        service.process_job(job.job_id)
    except RetryableProviderError:
        pass
    else:
        raise AssertionError("expected RetryableProviderError")

    stored = repository.get(job.job_id)
    assert stored.status == JobStatus.QUEUED
    assert stored.completed_at is None
    assert stored.started_at is None
    assert stored.error == "temporary failure"


def test_non_retryable_failure_persists_job_as_failed(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")

    repository = InMemoryJobRepository()
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
    service = TranscriptionService(
        RuntimeDependencies(
            settings=settings,
            repository=repository,
            artifact_store=store,
            runtime_state=runtime_state,
            router=ProviderRouter(["groq"]),
            remote_providers={"groq": FakeProvider("groq", error=NonRetryableProviderError("bad input"))},
            fallback_provider=None,
            inspector=FakeInspector(),
            chunker=AudioChunker(FakeInspector()),
        )
    )
    job = TranscriptionJob(
        job_id="job-terminal",
        status=JobStatus.QUEUED,
        payload=JobPayload(
            filename=audio_path.name,
            content_type="audio/wav",
            source_path=str(audio_path),
        ),
    )
    repository.create(job)

    try:
        service.process_job(job.job_id)
    except NonRetryableProviderError:
        pass
    else:
        raise AssertionError("expected NonRetryableProviderError")

    stored = repository.get(job.job_id)
    assert stored.status == JobStatus.FAILED
    assert stored.completed_at is not None
    assert stored.error == "bad input"
