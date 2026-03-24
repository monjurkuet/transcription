import time

import pytest
import redis

from audio_transcript.infra.queue import RedisQueueBackend
from audio_transcript.infra.runtime_state import RedisRuntimeState

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def redis_container():
    testcontainers = pytest.importorskip("testcontainers.redis")
    try:
        with testcontainers.RedisContainer("redis:7-alpine") as container:
            yield container
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Redis integration tests unavailable: {exc}")


@pytest.fixture
def redis_client(redis_container):
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    client = redis.Redis(host=host, port=int(port), decode_responses=True)
    client.flushdb()
    yield client
    client.flushdb()


def test_redis_queue_backend_round_trip(redis_client):
    queue = RedisQueueBackend(redis_client, "test:jobs")
    queue.enqueue("job-1")
    queue.enqueue("job-2")
    assert queue.dequeue(timeout=1) == "job-1"
    queue.requeue("job-1", retry_count=1)
    assert queue.get_retry_count("job-1") == 1
    assert queue.dequeue(timeout=1) == "job-2"
    assert queue.dequeue(timeout=1) == "job-1"
    queue.move_to_dlq("job-1", "boom", retry_count=1)
    queue.clear_retry_state("job-1")
    dlq = queue.get_dlq_jobs(limit=10)
    assert dlq[0]["job_id"] == "job-1"
    assert queue.healthcheck()["queue"] == "ok"


def test_redis_runtime_state_lock_and_cooldown(redis_client):
    state = RedisRuntimeState(redis_client)
    assert state.acquire_job_lock("job-1", ttl_seconds=1) is True
    assert state.acquire_job_lock("job-1", ttl_seconds=1) is False
    state.release_job_lock("job-1")
    assert state.acquire_job_lock("job-1", ttl_seconds=1) is True
    state.set_provider_cooldown("groq", "key-1", 1, "rate limit")
    assert state.get_provider_error("groq", "key-1") == "rate limit"
    assert state.get_provider_cooldown("groq", "key-1") is not None
    time.sleep(1.1)
    assert state.get_provider_cooldown("groq", "key-1") is None
    assert state.healthcheck()["runtime_state"] == "ok"
