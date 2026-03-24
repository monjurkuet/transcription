"""Background worker entrypoint."""

from __future__ import annotations

import logging
import time
from typing import Callable

from ..api.app import build_runtime
from ..config import Settings
from ..domain.errors import JobNotFoundError, RetryableProviderError
from ..domain.models import JobStatus, utcnow
from ..logging_utils import clear_job_context, set_job_context


BASE_BACKOFF_SEC = 2


def calculate_backoff(attempt: int) -> int:
    """Return the retry backoff in seconds for a given attempt number."""
    return BASE_BACKOFF_SEC * (2 ** (attempt - 1))


def _mark_job_failed(runtime: dict, job_id: str, error: str) -> None:
    """Persist a terminal failure for a job that cannot continue."""
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
    """Process one dequeued job and apply retry or DLQ policy."""
    logger = logging.getLogger("audio_transcript.worker")
    queue = runtime["queue"]
    service = runtime["service"]
    settings = runtime["settings"]
    job_token = set_job_context(job_id)

    try:
        service.process_job(job_id)
        queue.clear_retry_state(job_id)
        logger.info("worker finished job successfully", extra={"event": "worker_job_succeeded"})
    except RetryableProviderError as exc:
        retry_count = queue.get_retry_count(job_id) + 1
        logger.warning(
            "worker saw retryable job failure",
            extra={
                "event": "worker_job_retryable_failure",
                "retry_count": retry_count,
                "max_retries": settings.provider_max_retries,
                "error": str(exc),
            },
        )
        if retry_count >= settings.provider_max_retries:
            queue.move_to_dlq(job_id, str(exc), retry_count)
            queue.clear_retry_state(job_id)
            _mark_job_failed(runtime, job_id, "Retry limit exceeded")
            logger.error(
                "worker moved job to dead-letter queue after exhausting retries",
                extra={
                    "event": "worker_job_dlq",
                    "retry_count": retry_count,
                    "error": str(exc),
                },
            )
            return
        backoff = calculate_backoff(retry_count)
        logger.info(
            "worker scheduled job retry",
            extra={"event": "worker_job_retry_scheduled", "retry_count": retry_count, "backoff_sec": backoff},
        )
        sleep_fn(backoff)
        queue.requeue(job_id, retry_count)
    except Exception as exc:
        retry_count = queue.get_retry_count(job_id)
        logger.exception(
            "worker saw terminal job failure",
            extra={"event": "worker_job_failed", "retry_count": retry_count, "error": str(exc)},
        )
        queue.move_to_dlq(job_id, str(exc), retry_count)
        queue.clear_retry_state(job_id)
        _mark_job_failed(runtime, job_id, str(exc))
    finally:
        clear_job_context(job_token)


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
