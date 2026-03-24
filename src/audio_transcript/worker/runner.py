"""Background worker entrypoint."""

from __future__ import annotations

import logging
import time

from ..api.app import build_runtime
from ..config import Settings


def run_worker_loop(settings: Settings | None = None, poll_interval_sec: int = 1) -> None:
    """Run the background job worker forever."""
    settings = settings or Settings.from_env()
    runtime = build_runtime(settings)
    logger = logging.getLogger("audio_transcript.worker")
    queue = runtime["queue"]
    service = runtime["service"]

    logger.info("worker started")
    while True:
        job_id = queue.dequeue(timeout=poll_interval_sec)
        if not job_id:
            continue
        try:
            service.process_job(job_id)
        except Exception:
            logger.exception("job %s failed", job_id)
            time.sleep(1)


def run_single_iteration(runtime: dict) -> bool:
    """Test helper that processes a single queued job if present."""
    job_id = runtime["queue"].dequeue(timeout=0)
    if not job_id:
        return False
    runtime["service"].process_job(job_id)
    return True
