"""Transcription orchestration."""

from __future__ import annotations

import logging
import mimetypes
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from ..logging_utils import clear_job_context, set_job_context
from .audio import AudioChunker, AudioInspector, merge_transcripts
from .router import ProviderRouter


@dataclass
class RuntimeDependencies:
    """Shared collaborators required to process transcription jobs."""

    settings: Settings
    repository: JobRepository
    artifact_store: TranscriptArtifactStore
    runtime_state: RuntimeState
    router: ProviderRouter
    remote_providers: Dict[str, TranscriptionProvider]
    fallback_provider: Optional[TranscriptionProvider]
    inspector: AudioInspector
    chunker: AudioChunker


class _TranscriptionExecutionError(Exception):
    """Internal wrapper that preserves attempts generated before failure."""

    def __init__(self, cause: Exception, attempts: List[ProviderAttempt]):
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts


class TranscriptionService:
    """Coordinate job lifecycle, provider routing, and result persistence."""

    def __init__(self, deps: RuntimeDependencies):
        """Create a service from the pre-built runtime dependency graph."""
        self.deps = deps
        self.logger = logging.getLogger("audio_transcript.transcription")

    def create_job(self, job: TranscriptionJob) -> None:
        """Persist a queued job before it is handed to the worker."""
        self.deps.repository.create(job)

    def process_job(self, job_id: str) -> TranscriptionJob:
        """Process one job from queued state through completion or failure."""
        job_token = set_job_context(job_id)
        if not self.deps.runtime_state.acquire_job_lock(job_id):
            raise ValidationError(f"Job is already being processed: {job_id}")
        job: Optional[TranscriptionJob] = None
        started = time.monotonic()
        try:
            self.logger.info("job processing started", extra={"event": "job_started"})
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

            transcript, attempts = self._transcribe(audio_path, job)
            job.attempts.extend(attempts)
            job.status = JobStatus.SUCCEEDED
            job.completed_at = utcnow()
            job.provider = transcript.provider
            job.model = transcript.model
            job.summary_text = transcript.text[:1000]
            job.segment_count = len(transcript.segments)
            result_path = self.deps.artifact_store.save_result(job, transcript, file_metadata)
            job.result_path = result_path
            self.deps.repository.save(job)
            self.logger.info(
                "job processing completed",
                extra={
                    "event": "job_succeeded",
                    "provider": transcript.provider,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "segment_count": len(transcript.segments),
                },
            )
            return job
        except _TranscriptionExecutionError as exc:
            if job is not None:
                job.attempts.extend(exc.attempts)
            if isinstance(exc.cause, RetryableProviderError):
                if job is not None:
                    job.status = JobStatus.QUEUED
                    job.started_at = None
                    job.completed_at = None
                    job.error = str(exc.cause)
                    self.deps.repository.save(job)
                self.logger.warning(
                    "job processing failed with retryable provider error",
                    extra={
                        "event": "job_retryable_failure",
                        "error": str(exc.cause),
                        "duration_ms": int((time.monotonic() - started) * 1000),
                    },
                )
                raise exc.cause
            if job is not None:
                job.status = JobStatus.FAILED
                job.completed_at = utcnow()
                job.error = str(exc.cause)
                self.deps.repository.save(job)
            self.logger.error(
                "job processing failed",
                extra={
                    "event": "job_failed",
                    "error": str(exc.cause),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
            raise exc.cause
        except RetryableProviderError as exc:
            if job is not None:
                job.status = JobStatus.QUEUED
                job.started_at = None
                job.completed_at = None
                job.error = str(exc)
                self.deps.repository.save(job)
            self.logger.warning(
                "job processing failed with retryable provider error",
                extra={
                    "event": "job_retryable_failure",
                    "error": str(exc),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
            raise
        except Exception as exc:
            if job is not None:
                job.status = JobStatus.FAILED
                job.completed_at = utcnow()
                job.error = str(exc)
                self.deps.repository.save(job)
            self.logger.exception(
                "job processing failed with terminal error",
                extra={
                    "event": "job_failed",
                    "error": str(exc),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
            raise
        finally:
            self.deps.runtime_state.release_job_lock(job_id)
            clear_job_context(job_token)

    def _transcribe(self, audio_path: Path, job: TranscriptionJob) -> Tuple[TranscriptResult, List[ProviderAttempt]]:
        """Transcribe one source file, chunking first when it exceeds size limits."""
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        chunk_duration = job.payload.chunk_duration_sec or self.deps.settings.chunk_duration_sec
        chunk_overlap = job.payload.chunk_overlap_sec or self.deps.settings.chunk_overlap_sec

        if file_size_mb <= self.deps.settings.max_file_size_mb:
            return self._transcribe_single_with_attempts(audio_path, job.payload.content_type, job.payload.model_override)

        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_paths = self.deps.chunker.chunk_audio(audio_path, Path(temp_dir), chunk_duration, chunk_overlap)
            chunk_results, attempts = self._transcribe_chunks_parallel(chunk_paths, job.payload.model_override)
        return merge_transcripts(chunk_results, chunk_overlap), attempts

    def _transcribe_single_with_attempts(
        self,
        audio_path: Path,
        content_type: str,
        model_override: Optional[str],
    ) -> Tuple[TranscriptResult, List[ProviderAttempt]]:
        """Try remote providers in router order and fallback locally if configured."""
        remote_order = self.deps.router.select_remote_order(
            {name: True for name in self.deps.remote_providers.keys()}
        )
        errors: List[str] = []
        attempts: List[ProviderAttempt] = []
        for provider_name in remote_order:
            provider = self.deps.remote_providers[provider_name]
            try:
                return self._run_provider(provider, audio_path, content_type, model_override, attempts), attempts
            except NonRetryableProviderError as exc:
                errors.append(str(exc))
                raise _TranscriptionExecutionError(exc, attempts) from exc
            except (RetryableProviderError, RuntimeError) as exc:
                errors.append(str(exc))
                continue

        if self.deps.fallback_provider:
            try:
                return (
                    self._run_provider(self.deps.fallback_provider, audio_path, content_type, model_override, attempts),
                    attempts,
                )
            except Exception as exc:
                errors.append(str(exc))

        failure = RetryableProviderError("; ".join(errors) or "No provider succeeded")
        raise _TranscriptionExecutionError(failure, attempts)

    def _transcribe_chunks_parallel(
        self,
        chunk_paths: List[Path],
        model_override: Optional[str],
    ) -> Tuple[List[TranscriptResult], List[ProviderAttempt]]:
        """Transcribe chunked audio concurrently and preserve attempt ordering."""
        if not chunk_paths:
            return [], []

        max_workers = max(1, min(self.deps.settings.max_parallel_chunks, len(chunk_paths)))
        results: Dict[int, TranscriptResult] = {}
        attempts_by_index: Dict[int, List[ProviderAttempt]] = {}
        failures: List[Tuple[int, Exception, List[ProviderAttempt]]] = []

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="chunk-transcribe") as executor:
            future_to_index = {
                executor.submit(self._transcribe_single_with_attempts, chunk_path, "audio/wav", model_override): index
                for index, chunk_path in enumerate(chunk_paths)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result, attempts = future.result()
                    results[index] = result
                    attempts_by_index[index] = attempts
                except _TranscriptionExecutionError as exc:
                    failures.append((index, exc.cause, exc.attempts))

        if failures:
            failures.sort(key=lambda item: item[0])
            ordered_attempts = self._flatten_attempts(attempts_by_index, failures)
            non_retryable = next((item for item in failures if isinstance(item[1], NonRetryableProviderError)), None)
            chosen = non_retryable or failures[0]
            raise _TranscriptionExecutionError(chosen[1], ordered_attempts)

        ordered_attempts = self._flatten_attempts(attempts_by_index)
        ordered_results = [results[index] for index in range(len(chunk_paths))]
        return ordered_results, ordered_attempts

    def _flatten_attempts(
        self,
        attempts_by_index: Dict[int, List[ProviderAttempt]],
        failures: Optional[List[Tuple[int, Exception, List[ProviderAttempt]]]] = None,
    ) -> List[ProviderAttempt]:
        """Merge successful and failed chunk attempts into one ordered audit trail."""
        ordered: List[ProviderAttempt] = []
        failure_map = {index: attempts for index, _, attempts in failures or []}
        for index in sorted(set(attempts_by_index) | set(failure_map)):
            ordered.extend(attempts_by_index.get(index, []))
            ordered.extend(failure_map.get(index, []))
        return ordered

    def _run_provider(
        self,
        provider: TranscriptionProvider,
        audio_path: Path,
        content_type: str,
        model_override: Optional[str],
        attempts: List[ProviderAttempt],
    ) -> TranscriptResult:
        """Execute one provider attempt and append normalized attempt metadata."""
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
            self.logger.info(
                "provider transcription succeeded",
                extra={
                    "event": "provider_attempt_succeeded",
                    "provider": provider.provider_name,
                    "duration_ms": int((time.monotonic() - start) * 1000),
                    "model": result.model,
                },
            )
            return result
        except NonRetryableProviderError as exc:
            attempt.error = str(exc)
            attempt.retryable = False
            self.logger.warning(
                "provider transcription failed",
                extra={
                    "event": "provider_attempt_failed",
                    "provider": provider.provider_name,
                    "duration_ms": int((time.monotonic() - start) * 1000),
                    "error": str(exc),
                    "retryable": False,
                },
            )
            raise
        except RetryableProviderError as exc:
            attempt.error = str(exc)
            attempt.retryable = True
            self.logger.warning(
                "provider transcription failed",
                extra={
                    "event": "provider_attempt_failed",
                    "provider": provider.provider_name,
                    "duration_ms": int((time.monotonic() - start) * 1000),
                    "error": str(exc),
                    "retryable": True,
                },
            )
            raise
        except Exception as exc:
            attempt.error = str(exc)
            attempt.retryable = False
            self.logger.warning(
                "provider transcription failed",
                extra={
                    "event": "provider_attempt_failed",
                    "provider": provider.provider_name,
                    "duration_ms": int((time.monotonic() - start) * 1000),
                    "error": str(exc),
                    "retryable": False,
                },
            )
            raise
        finally:
            attempt.finished_at = utcnow()
            attempt.latency_ms = int((time.monotonic() - start) * 1000)
            attempts.append(attempt)
