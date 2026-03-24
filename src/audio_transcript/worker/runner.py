"""Background worker entrypoint."""

from __future__ import annotations

import logging
import time
from typing import Callable

from ..api.app import build_runtime
from ..config import Settings
from ..domain.errors import JobNotFoundError, RetryableProviderError
from ..domain.models import JobStatus, utcnow


BASE_BACKOFF_SEC = 2


def calculate_backoff(attempt: int) -> int:
    """Return the retry backoff in seconds for a given attempt number."""
    return BASE_BACKOFF_SEC * (2 ** (attempt - 1))


def _mark_job_failed(runtime: dict, job_id: str, error: str) -> None:
    repository = runtime["repository"]
    try:
        job = repository.get(job_id)
    except JobNotFoundError:
        return
    job.status = JobStatus.FAILED
    job.completed_at = utcnow()
    job.error = error
    repository.save(job)


def _handle_job(runtime: dict, job_id: str, sleep_fn: Callable[[float], None]) -> None:
    logger = logging.getLogger("audio_transcript.worker")
    queue = runtime["queue"]
    service = runtime["service"]
    settings = runtime["settings"]

    try:
        service.process_job(job_id)
        queue.clear_retry_state(job_id)
    except RetryableProviderError as exc:
        retry_count = queue.get_retry_count(job_id) + 1
        logger.exception("job %s failed with retryable error (attempt %d/%d)", job_id, retry_count, settings.provider_max_retries)
        if retry_count >= settings.provider_max_retries:
            queue.move_to_dlq(job_id, str(exc), retry_count)
            queue.clear_retry_state(job_id)
            _mark_job_failed(runtime, job_id, "Retry limit exceeded")
            return
        backoff = calculate_backoff(retry_count)
        logger.info("job %s will retry after %ss", job_id, backoff)
        sleep_fn(backoff)
        queue.requeue(job_id, retry_count)
    except Exception as exc:
        retry_count = queue.get_retry_count(job_id)
        logger.exception("job %s failed with terminal error", job_id)
        queue.move_to_dlq(job_id, str(exc), retry_count)
        queue.clear_retry_state(job_id)
        _mark_job_failed(runtime, job_id, str(exc))


def run_worker_loop(
    settings: Settings | None = None,
    poll_interval_sec: int = 1,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Run the background job worker forever."""
    settings = settings or Settings.from_env()
    runtime = build_runtime(settings)
    logger = logging.getLogger("audio_transcript.worker")
    queue = runtime["queue"]

    logger.info("worker started")
    while True:
        job_id = queue.dequeue(timeout=poll_interval_sec)
        if not job_id:
            continue
        _handle_job(runtime, job_id, sleep_fn)


def run_single_iteration(runtime: dict, sleep_fn: Callable[[float], None] = time.sleep) -> bool:
    """Test helper that processes a single queued job if present."""
    job_id = runtime["queue"].dequeue(timeout=0)
    if not job_id:
        return False
    _handle_job(runtime, job_id, sleep_fn)
    return True
