import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_transcript.api.app import create_app
from audio_transcript.config import Settings
from audio_transcript.domain.models import FileMetadata, TranscriptResult, TranscriptSegment
from audio_transcript.infra.queue import InMemoryQueueBackend
from audio_transcript.infra.repository import InMemoryJobRepository
from audio_transcript.infra.runtime_state import InMemoryRuntimeState
from audio_transcript.infra.storage import TranscriptArtifactStore
from audio_transcript.services.audio import AudioChunker, AudioInspector
from audio_transcript.services.router import ProviderRouter
from audio_transcript.services.transcription import RuntimeDependencies, TranscriptionService


class FakeInspector(AudioInspector):
    def get_file_metadata(self, file_path):
        return FileMetadata(
            filename=file_path.name,
            path=str(file_path),
            size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            duration=1.0,
            format=file_path.suffix.lstrip(".") or "wav",
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
    def __init__(self, name, outcomes=None, result=None, error=None):
        self.provider_name = name
        self.outcomes = list(outcomes or [])
        self.result = result
        self.error = error

    def transcribe(self, audio_path, content_type, model_override=None):
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
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
    return app, {
        "settings": settings,
        "repository": repository,
        "queue": queue,
        "service": service,
        "artifact_store": store,
    }


def ffmpeg_available() -> bool:
    import subprocess

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def pytest_collection_modifyitems(config, items):
    markexpr = (config.option.markexpr or "").replace(" ", "")
    run_integration = os.getenv("RUN_INTEGRATION") == "1" or "integration" in markexpr
    run_ffmpeg = os.getenv("RUN_FFMPEG") == "1" or "ffmpeg" in markexpr

    integration_skip = pytest.mark.skip(reason="integration tests are opt-in; use RUN_INTEGRATION=1 or -m integration")
    ffmpeg_skip = pytest.mark.skip(reason="ffmpeg tests are opt-in; use RUN_FFMPEG=1 or -m ffmpeg")

    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(integration_skip)
        if "ffmpeg" in item.keywords and not run_ffmpeg:
            item.add_marker(ffmpeg_skip)
