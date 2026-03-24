"""Transient runtime state backed by Redis or memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import redis


class RuntimeState(ABC):
    """Ephemeral runtime coordination."""

    @abstractmethod
    def acquire_job_lock(self, job_id: str, ttl_seconds: int = 3600) -> bool:
        """Acquire an in-flight lock for a job."""

    @abstractmethod
    def release_job_lock(self, job_id: str) -> None:
        """Release an in-flight lock for a job."""

    @abstractmethod
    def set_provider_cooldown(self, provider: str, key_id: str, seconds: int, error: str) -> None:
        """Store provider cooldown state."""

    @abstractmethod
    def get_provider_cooldown(self, provider: str, key_id: str) -> Optional[datetime]:
        """Fetch provider cooldown state."""

    @abstractmethod
    def get_provider_error(self, provider: str, key_id: str) -> Optional[str]:
        """Fetch last provider error."""

    @abstractmethod
    def healthcheck(self) -> Dict[str, str]:
        """Readiness info."""


class RedisRuntimeState(RuntimeState):
    """Redis-backed locks and cooldown state."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def acquire_job_lock(self, job_id: str, ttl_seconds: int = 3600) -> bool:
        return bool(self.redis.set(f"audio-transcript:lock:{job_id}", "1", nx=True, ex=ttl_seconds))

    def release_job_lock(self, job_id: str) -> None:
        self.redis.delete(f"audio-transcript:lock:{job_id}")

    def set_provider_cooldown(self, provider: str, key_id: str, seconds: int, error: str) -> None:
        self.redis.set(
            f"audio-transcript:cooldown:{provider}:{key_id}",
            datetime.now(timezone.utc).isoformat(),
            ex=seconds,
        )
        self.redis.set(
            f"audio-transcript:error:{provider}:{key_id}",
            error,
            ex=max(seconds, 3600),
        )

    def get_provider_cooldown(self, provider: str, key_id: str) -> Optional[datetime]:
        ttl = self.redis.ttl(f"audio-transcript:cooldown:{provider}:{key_id}")
        if ttl is None or ttl <= 0:
            return None
        return datetime.now(timezone.utc) + timedelta(seconds=ttl)

    def get_provider_error(self, provider: str, key_id: str) -> Optional[str]:
        value = self.redis.get(f"audio-transcript:error:{provider}:{key_id}")
        return value if isinstance(value, str) else None

    def healthcheck(self) -> Dict[str, str]:
        self.redis.ping()
        return {"runtime_state": "ok"}


class InMemoryRuntimeState(RuntimeState):
    """In-memory state for tests."""

    def __init__(self):
        self.locks: set[str] = set()
        self.cooldowns: Dict[tuple[str, str], datetime] = {}
        self.errors: Dict[tuple[str, str], str] = {}

    def acquire_job_lock(self, job_id: str, ttl_seconds: int = 3600) -> bool:
        if job_id in self.locks:
            return False
        self.locks.add(job_id)
        return True

    def release_job_lock(self, job_id: str) -> None:
        self.locks.discard(job_id)

    def set_provider_cooldown(self, provider: str, key_id: str, seconds: int, error: str) -> None:
        self.cooldowns[(provider, key_id)] = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        self.errors[(provider, key_id)] = error

    def get_provider_cooldown(self, provider: str, key_id: str) -> Optional[datetime]:
        value = self.cooldowns.get((provider, key_id))
        if value and value <= datetime.now(timezone.utc):
            self.cooldowns.pop((provider, key_id), None)
            return None
        return value

    def get_provider_error(self, provider: str, key_id: str) -> Optional[str]:
        return self.errors.get((provider, key_id))

    def healthcheck(self) -> Dict[str, str]:
        return {"runtime_state": "ok"}
