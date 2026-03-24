"""Queue backends for background jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, Optional

import redis


class QueueBackend(ABC):
    """Queue interface."""

    @abstractmethod
    def enqueue(self, job_id: str) -> None:
        """Enqueue a job."""

    @abstractmethod
    def dequeue(self, timeout: int = 1) -> Optional[str]:
        """Dequeue a job id."""

    @abstractmethod
    def healthcheck(self) -> Dict[str, str]:
        """Readiness info."""


class RedisQueueBackend(QueueBackend):
    """Redis list-based queue."""

    def __init__(self, redis_client: redis.Redis, queue_name: str):
        self.redis = redis_client
        self.queue_name = queue_name

    def enqueue(self, job_id: str) -> None:
        self.redis.rpush(self.queue_name, job_id)

    def dequeue(self, timeout: int = 1) -> Optional[str]:
        result = self.redis.blpop(self.queue_name, timeout=timeout)
        if result is None:
            return None
        _, job_id = result
        return job_id.decode("utf-8") if isinstance(job_id, bytes) else job_id

    def healthcheck(self) -> Dict[str, str]:
        self.redis.ping()
        return {"queue": "ok"}


class InMemoryQueueBackend(QueueBackend):
    """In-memory queue for tests."""

    def __init__(self):
        self.items: Deque[str] = deque()

    def enqueue(self, job_id: str) -> None:
        self.items.append(job_id)

    def dequeue(self, timeout: int = 1) -> Optional[str]:
        if not self.items:
            return None
        return self.items.popleft()

    def healthcheck(self) -> Dict[str, str]:
        return {"memory_queue": "ok"}
