import io

from audio_transcript.api.app import create_app
from audio_transcript.config import Settings
from audio_transcript.domain.models import TranscriptResult, TranscriptSegment
from audio_transcript.infra.queue import InMemoryQueueBackend
from audio_transcript.infra.repository import InMemoryJobRepository
from audio_transcript.infra.runtime_state import InMemoryRuntimeState
from audio_transcript.infra.storage import TranscriptArtifactStore
from audio_transcript.services.audio import AudioChunker
from audio_transcript.services.router import ProviderRouter
from audio_transcript.services.transcription import RuntimeDependencies, TranscriptionService
from audio_transcript.worker.runner import run_single_iteration


class FakeInspector:
    def get_file_metadata(self, file_path):
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

    def get_duration(self, audio_path):
        return 1.0


class FakeChunker(AudioChunker):
    def __init__(self):
        super().__init__(FakeInspector())


class FakeProvider:
    def __init__(self, name):
        self.provider_name = name

    def transcribe(self, audio_path, content_type, model_override=None):
        return TranscriptResult(
            text=f"{self.provider_name} text",
            segments=[TranscriptSegment(start=0.0, end=1.0, text=f"{self.provider_name} text")],
            provider=self.provider_name,
            model=model_override,
        )

    def status(self):
        return {"provider": self.provider_name}


def build_test_app(tmp_path):
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
        mistral_api_keys=["ms_1"],
    )
    providers = {"groq": FakeProvider("groq"), "mistral": FakeProvider("mistral")}
    service = TranscriptionService(
        RuntimeDependencies(
            settings=settings,
            repository=repository,
            artifact_store=store,
            runtime_state=runtime_state,
            router=ProviderRouter(["groq", "mistral"]),
            remote_providers=providers,
            fallback_provider=FakeProvider("whisper_cpp"),
            inspector=FakeInspector(),
            chunker=FakeChunker(),
        )
    )
    app = create_app(
        settings,
        repository=repository,
        queue=queue,
        artifact_store=store,
        runtime_state=runtime_state,
        service=service,
        providers=providers,
        fallback_provider=FakeProvider("whisper_cpp"),
    )
    return app, {"repository": repository, "queue": queue, "service": service, "artifact_store": store}


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
