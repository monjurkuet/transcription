"""Queue backends for background jobs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Optional

import redis

from ..domain.models import utcnow


class QueueBackend(ABC):
    """Queue interface."""

    @abstractmethod
    def enqueue(self, job_id: str) -> None:
        """Enqueue a job."""

    @abstractmethod
    def dequeue(self, timeout: int = 1) -> Optional[str]:
        """Dequeue a job id."""

    @abstractmethod
    def get_retry_count(self, job_id: str) -> int:
        """Return the retry count for a job."""

    @abstractmethod
    def requeue(self, job_id: str, retry_count: int) -> None:
        """Requeue a job after recording its retry count."""

    @abstractmethod
    def clear_retry_state(self, job_id: str) -> None:
        """Clear retry metadata for a job."""

    @abstractmethod
    def move_to_dlq(self, job_id: str, error: str, retry_count: int) -> None:
        """Persist terminal failure metadata in the dead-letter queue."""

    @abstractmethod
    def get_dlq_jobs(self, limit: int = 100) -> List[Dict[str, str]]:
        """Return dead-letter queue items."""

    @abstractmethod
    def healthcheck(self) -> Dict[str, str]:
        """Readiness info."""


class RedisQueueBackend(QueueBackend):
    """Redis list-based queue."""

    def __init__(self, redis_client: redis.Redis, queue_name: str):
        self.redis = redis_client
        self.queue_name = queue_name
        self.retry_name = f"{queue_name}:retries"
        self.dlq_name = f"{queue_name}:dlq"

    def enqueue(self, job_id: str) -> None:
        self.redis.rpush(self.queue_name, job_id)

    def dequeue(self, timeout: int = 1) -> Optional[str]:
        result = self.redis.blpop(self.queue_name, timeout=timeout)
        if result is None:
            return None
        _, job_id = result
        return job_id.decode("utf-8") if isinstance(job_id, bytes) else job_id

    def get_retry_count(self, job_id: str) -> int:
        value = self.redis.hget(self.retry_name, job_id)
        if value is None:
            return 0
        return int(value)

    def requeue(self, job_id: str, retry_count: int) -> None:
        pipeline = self.redis.pipeline()
        pipeline.hset(self.retry_name, job_id, retry_count)
        pipeline.rpush(self.queue_name, job_id)
        pipeline.execute()

    def clear_retry_state(self, job_id: str) -> None:
        self.redis.hdel(self.retry_name, job_id)

    def move_to_dlq(self, job_id: str, error: str, retry_count: int) -> None:
        payload = json.dumps(
            {
                "job_id": job_id,
                "error": error,
                "retry_count": retry_count,
                "moved_at": utcnow().isoformat(),
            }
        )
        self.redis.lpush(self.dlq_name, payload)

    def get_dlq_jobs(self, limit: int = 100) -> List[Dict[str, str]]:
        items = self.redis.lrange(self.dlq_name, 0, max(limit - 1, 0))
        jobs: List[Dict[str, str]] = []
        for item in items:
            value = item.decode("utf-8") if isinstance(item, bytes) else item
            jobs.append(json.loads(value))
        return jobs

    def healthcheck(self) -> Dict[str, str]:
        self.redis.ping()
        return {"queue": "ok"}


class InMemoryQueueBackend(QueueBackend):
    """In-memory queue for tests."""

    def __init__(self):
        self.items: Deque[str] = deque()
        self.retry_counts: Dict[str, int] = {}
        self.dlq: List[Dict[str, str]] = []

    def enqueue(self, job_id: str) -> None:
        self.items.append(job_id)

    def dequeue(self, timeout: int = 1) -> Optional[str]:
        if not self.items:
            return None
        return self.items.popleft()

    def get_retry_count(self, job_id: str) -> int:
        return self.retry_counts.get(job_id, 0)

    def requeue(self, job_id: str, retry_count: int) -> None:
        self.retry_counts[job_id] = retry_count
        self.items.append(job_id)

    def clear_retry_state(self, job_id: str) -> None:
        self.retry_counts.pop(job_id, None)

    def move_to_dlq(self, job_id: str, error: str, retry_count: int) -> None:
        self.dlq.insert(
            0,
            {
                "job_id": job_id,
                "error": error,
                "retry_count": retry_count,
                "moved_at": utcnow().isoformat(),
            },
        )

    def get_dlq_jobs(self, limit: int = 100) -> List[Dict[str, str]]:
        return self.dlq[:limit]

    def healthcheck(self) -> Dict[str, str]:
        return {"memory_queue": "ok"}
