"""Transcription orchestration."""

from __future__ import annotations

import logging
import mimetypes
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Settings
from ..domain.errors import NonRetryableProviderError, RetryableProviderError, ValidationError
from ..domain.models import (
    JobStatus,
    ProviderAttempt,
    TranscriptionJob,
    TranscriptResult,
    is_supported_audio_file,
    utcnow,
)
from ..infra.providers.base import TranscriptionProvider
from ..infra.repository import JobRepository
from ..infra.runtime_state import RuntimeState
from ..infra.storage import TranscriptArtifactStore
from .audio import AudioChunker, AudioInspector, merge_transcripts
from .router import ProviderRouter


@dataclass
class RuntimeDependencies:
    """Shared runtime dependencies."""

    settings: Settings
    repository: JobRepository
    artifact_store: TranscriptArtifactStore
    runtime_state: RuntimeState
    router: ProviderRouter
    remote_providers: Dict[str, TranscriptionProvider]
    fallback_provider: Optional[TranscriptionProvider]
    inspector: AudioInspector
    chunker: AudioChunker


class TranscriptionService:
    """Coordinates job execution."""

    def __init__(self, deps: RuntimeDependencies):
        self.deps = deps
        self.logger = logging.getLogger("audio_transcript.transcription")

    def create_job(self, job: TranscriptionJob) -> None:
        self.deps.repository.create(job)

    def process_job(self, job_id: str) -> TranscriptionJob:
        if not self.deps.runtime_state.acquire_job_lock(job_id):
            raise ValidationError(f"Job is already being processed: {job_id}")
        job: Optional[TranscriptionJob] = None
        try:
            job = self.deps.repository.get(job_id)
            audio_path = Path(job.payload.source_path)
            if not audio_path.exists():
                raise ValidationError(f"Source artifact does not exist: {audio_path}")
            if not is_supported_audio_file(audio_path):
                raise ValidationError(f"Unsupported audio file: {audio_path.name}")

            job.status = JobStatus.PROCESSING
            job.started_at = utcnow()
            self.deps.repository.save(job)

            file_metadata = self.deps.inspector.get_file_metadata(audio_path)
            job.file_metadata = file_metadata

            transcript = self._transcribe(audio_path, job)
            job.status = JobStatus.SUCCEEDED
            job.completed_at = utcnow()
            job.provider = transcript.provider
            job.model = transcript.model
            job.summary_text = transcript.text[:1000]
            job.segment_count = len(transcript.segments)
            result_path = self.deps.artifact_store.save_result(job, transcript, file_metadata)
            job.result_path = result_path
            self.deps.repository.save(job)
            return job
        except Exception as exc:
            if job is not None:
                job.status = JobStatus.FAILED
                job.completed_at = utcnow()
                job.error = str(exc)
                self.deps.repository.save(job)
            raise
        finally:
            self.deps.runtime_state.release_job_lock(job_id)

    def _transcribe(self, audio_path: Path, job: TranscriptionJob) -> TranscriptResult:
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        chunk_duration = job.payload.chunk_duration_sec or self.deps.settings.chunk_duration_sec
        chunk_overlap = job.payload.chunk_overlap_sec or self.deps.settings.chunk_overlap_sec

        if file_size_mb <= self.deps.settings.max_file_size_mb:
            return self._transcribe_single(audio_path, job.payload.content_type, job.payload.model_override, job)

        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_paths = self.deps.chunker.chunk_audio(audio_path, Path(temp_dir), chunk_duration, chunk_overlap)
            chunk_results = [
                self._transcribe_single(chunk_path, "audio/wav", job.payload.model_override, job)
                for chunk_path in chunk_paths
            ]
        return merge_transcripts(chunk_results, chunk_overlap)

    def _transcribe_single(
        self,
        audio_path: Path,
        content_type: str,
        model_override: Optional[str],
        job: TranscriptionJob,
    ) -> TranscriptResult:
        remote_order = self.deps.router.select_remote_order(
            {name: True for name in self.deps.remote_providers.keys()}
        )
        errors: List[str] = []
        for provider_name in remote_order:
            provider = self.deps.remote_providers[provider_name]
            try:
                return self._run_provider(provider, audio_path, content_type, model_override, job)
            except NonRetryableProviderError as exc:
                errors.append(str(exc))
                raise
            except (RetryableProviderError, RuntimeError) as exc:
                errors.append(str(exc))
                continue

        if self.deps.fallback_provider:
            try:
                return self._run_provider(self.deps.fallback_provider, audio_path, content_type, model_override, job)
            except Exception as exc:
                errors.append(str(exc))

        raise RetryableProviderError("; ".join(errors) or "No provider succeeded")

    def _run_provider(
        self,
        provider: TranscriptionProvider,
        audio_path: Path,
        content_type: str,
        model_override: Optional[str],
        job: TranscriptionJob,
    ) -> TranscriptResult:
        attempt = ProviderAttempt(
            provider=provider.provider_name,
            started_at=utcnow(),
            model=model_override,
        )
        start = time.monotonic()
        try:
            result = provider.transcribe(audio_path, content_type, model_override=model_override)
            attempt.success = True
            attempt.model = result.model
            return result
        except NonRetryableProviderError as exc:
            attempt.error = str(exc)
            attempt.retryable = False
            raise
        except RetryableProviderError as exc:
            attempt.error = str(exc)
            attempt.retryable = True
            raise
        except Exception as exc:
            attempt.error = str(exc)
            attempt.retryable = False
            raise
        finally:
            attempt.finished_at = utcnow()
            attempt.latency_ms = int((time.monotonic() - start) * 1000)
            job.attempts.append(attempt)
            self.deps.repository.save(job)
