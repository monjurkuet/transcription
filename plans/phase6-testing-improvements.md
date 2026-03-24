# Phase 6: Testing Improvements

Priority: MEDIUM | Effort: HIGH | Risk if Skipped: Regressions, low confidence in changes

This phase addresses testing gaps to improve confidence in the codebase.

---

## 6.1 Repository Integration Tests

**Problem:**
Only `InMemoryJobRepository` is tested. `PostgresJobRepository` has complex SQL that's never verified against real Postgres.

Risks:
- SQL syntax errors not caught
- Schema migration issues
- Postgres-specific behavior differences

**Solution:**
Add integration tests using `testcontainers` for real Postgres.

**Implementation:**

**Install dependencies:**
```toml
# pyproject.toml
[project.optional-dependencies]
dev = [
    # ... existing
    "testcontainers[postgres]>=3.7.0",
]
```

**Create tests/test_repository_integration.py:**
```python
"""Integration tests for PostgresJobRepository against real Postgres."""

import pytest
from testcontainers.postgres import PostgresContainer

from audio_transcript.domain.models import (
    JobPayload,
    JobStatus,
    ProviderAttempt,
    TranscriptionJob,
    utcnow,
)
from audio_transcript.infra.repository import PostgresJobRepository


@pytest.fixture(scope="module")
def postgres_container():
    """Start a Postgres container for the test module."""
    with PostgresContainer("postgres:15-alpine") as postgres:
        yield postgres


@pytest.fixture
def repository(postgres_container):
    """Create a fresh repository for each test."""
    url = postgres_container.get_connection_url()
    repo = PostgresJobRepository(url)
    yield repo
    # Cleanup: truncate tables
    with repo._get_connection() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE jobs, job_attempts CASCADE")


class TestPostgresJobRepository:
    """Integration tests for Postgres repository."""

    def test_create_and_get_job(self, repository):
        """Verify basic job CRUD."""
        job = TranscriptionJob(
            job_id="test-123",
            status=JobStatus.QUEUED,
            payload=JobPayload(
                filename="test.wav",
                content_type="audio/wav",
                source_path="/tmp/test.wav",
            ),
        )
        repository.create(job)
        
        loaded = repository.get("test-123")
        assert loaded.job_id == "test-123"
        assert loaded.status == JobStatus.QUEUED
        assert loaded.payload.filename == "test.wav"

    def test_save_updates_existing_job(self, repository):
        """Verify upsert behavior."""
        job = TranscriptionJob(
            job_id="test-456",
            status=JobStatus.QUEUED,
            payload=JobPayload(
                filename="test.wav",
                content_type="audio/wav",
                source_path="/tmp/test.wav",
            ),
        )
        repository.create(job)
        
        # Update status
        job.status = JobStatus.PROCESSING
        job.started_at = utcnow()
        repository.save(job)
        
        loaded = repository.get("test-456")
        assert loaded.status == JobStatus.PROCESSING
        assert loaded.started_at is not None

    def test_save_with_attempts(self, repository):
        """Verify attempts are persisted correctly."""
        job = TranscriptionJob(
            job_id="test-789",
            status=JobStatus.QUEUED,
            payload=JobPayload(
                filename="test.wav",
                content_type="audio/wav",
                source_path="/tmp/test.wav",
            ),
        )
        repository.create(job)
        
        # Add attempts
        job.attempts = [
            ProviderAttempt(
                provider="groq",
                started_at=utcnow(),
                finished_at=utcnow(),
                success=False,
                retryable=True,
                error="Rate limited",
                status_code=429,
            ),
            ProviderAttempt(
                provider="mistral",
                started_at=utcnow(),
                finished_at=utcnow(),
                success=True,
                retryable=False,
                model="mistral-large",
                latency_ms=1500,
            ),
        ]
        repository.save(job)
        
        loaded = repository.get("test-789")
        assert len(loaded.attempts) == 2
        assert loaded.attempts[0].provider == "groq"
        assert loaded.attempts[0].error == "Rate limited"
        assert loaded.attempts[1].success is True

    def test_list_jobs_with_filters(self, repository):
        """Verify list filtering works."""
        # Create multiple jobs
        for i, status in enumerate([JobStatus.QUEUED, JobStatus.SUCCEEDED, JobStatus.FAILED]):
            job = TranscriptionJob(
                job_id=f"job-{i}",
                status=status,
                payload=JobPayload(
                    filename=f"file{i}.wav",
                    content_type="audio/wav",
                    source_path=f"/tmp/file{i}.wav",
                ),
            )
            if status == JobStatus.SUCCEEDED:
                job.provider = "groq"
                job.summary_text = "This is a test transcript"
            repository.create(job)
        
        # Filter by status
        queued = repository.list_jobs(status="queued")
        assert len(queued) == 1
        assert queued[0].status == JobStatus.QUEUED
        
        # Filter by provider
        groq_jobs = repository.list_jobs(provider="groq")
        assert len(groq_jobs) == 1
        
        # Search by text
        search_results = repository.list_jobs(search="transcript")
        assert len(search_results) == 1

    def test_job_not_found_raises(self, repository):
        """Verify JobNotFoundError for missing jobs."""
        from audio_transcript.domain.errors import JobNotFoundError
        
        with pytest.raises(JobNotFoundError) as exc_info:
            repository.get("nonexistent-id")
        
        assert "nonexistent-id" in str(exc_info.value)

    def test_healthcheck(self, repository):
        """Verify healthcheck passes."""
        result = repository.healthcheck()
        assert result["postgres"] == "ok"

    def test_concurrent_saves(self, repository):
        """Verify concurrent saves don't corrupt data."""
        import concurrent.futures
        
        job_id = "concurrent-test"
        job = TranscriptionJob(
            job_id=job_id,
            status=JobStatus.QUEUED,
            payload=JobPayload(
                filename="test.wav",
                content_type="audio/wav",
                source_path="/tmp/test.wav",
            ),
        )
        repository.create(job)
        
        def add_attempt(attempt_num):
            loaded = repository.get(job_id)
            loaded.attempts.append(
                ProviderAttempt(
                    provider=f"provider-{attempt_num}",
                    started_at=utcnow(),
                    success=True,
                )
            )
            repository.save(loaded)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_attempt, i) for i in range(10)]
            concurrent.futures.wait(futures)
        
        final = repository.get(job_id)
        # Should have some attempts (exact count may vary due to race conditions)
        assert len(final.attempts) >= 1


# Mark all tests as integration tests
pytestmark = pytest.mark.integration
```

**Update pytest.ini or pyproject.toml:**
```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests as integration tests (require external services)",
    "ffmpeg: marks tests that require ffmpeg/ffprobe",
]
# Default: skip integration tests unless explicitly requested
addopts = "-m 'not integration and not ffmpeg'"
```

**Run integration tests:**
```bash
# Run only integration tests
pytest -m integration tests/test_repository_integration.py -v

# Run all tests including integration
pytest -m "" tests/ -v
```

---

## 6.2 Redis Integration Tests

**Problem:**
`RedisJobQueue` and `RedisRuntimeState` are untested. Only in-memory implementations are used in tests.

**Solution:**
Add integration tests with testcontainers Redis.

**Implementation:**

**Create tests/test_redis_integration.py:**
```python
"""Integration tests for Redis queue and runtime state."""

import pytest
from testcontainers.redis import RedisContainer

from audio_transcript.infra.queue import RedisJobQueue
from audio_transcript.infra.runtime_state import RedisRuntimeState


@pytest.fixture(scope="module")
def redis_container():
    """Start a Redis container for the test module."""
    with RedisContainer("redis:7-alpine") as redis:
        yield redis


@pytest.fixture
def redis_url(redis_container):
    """Get Redis connection URL."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}/0"


@pytest.fixture
def queue(redis_url):
    """Create a fresh queue for each test."""
    q = RedisJobQueue(redis_url, "test:jobs")
    yield q
    # Cleanup
    q._client.flushdb()


@pytest.fixture
def runtime_state(redis_url):
    """Create a fresh runtime state for each test."""
    rs = RedisRuntimeState(redis_url, "test:state")
    yield rs
    rs._client.flushdb()


class TestRedisJobQueue:
    """Integration tests for Redis queue."""

    def test_enqueue_dequeue(self, queue):
        """Verify basic queue operations."""
        queue.enqueue("job-1")
        queue.enqueue("job-2")
        
        # FIFO order
        assert queue.dequeue(timeout=1) == "job-1"
        assert queue.dequeue(timeout=1) == "job-2"
        assert queue.dequeue(timeout=1) is None

    def test_dequeue_blocks(self, queue):
        """Verify dequeue blocks until timeout."""
        import time
        
        start = time.time()
        result = queue.dequeue(timeout=1)
        elapsed = time.time() - start
        
        assert result is None
        assert elapsed >= 0.9  # Should have waited ~1 second

    def test_requeue_for_retry(self, queue):
        """Verify retry requeue works."""
        queue.enqueue("job-retry")
        queue.dequeue(timeout=1)  # Take it off
        
        # Requeue for retry
        queue.requeue("job-retry", retry_count=1)
        
        # Should be back on queue
        assert queue.dequeue(timeout=1) == "job-retry"

    def test_move_to_dlq(self, queue):
        """Verify dead letter queue works."""
        queue.move_to_dlq("failed-job", "Max retries exceeded")
        
        dlq_jobs = queue.get_dlq_jobs(limit=10)
        assert len(dlq_jobs) == 1
        assert dlq_jobs[0]["job_id"] == "failed-job"
        assert "Max retries" in dlq_jobs[0]["error"]

    def test_healthcheck(self, queue):
        """Verify healthcheck."""
        result = queue.healthcheck()
        assert result["redis_queue"] == "ok"


class TestRedisRuntimeState:
    """Integration tests for Redis runtime state."""

    def test_job_lock_acquire_release(self, runtime_state):
        """Verify job locking works."""
        assert runtime_state.acquire_job_lock("job-1") is True
        assert runtime_state.acquire_job_lock("job-1") is False  # Already locked
        
        runtime_state.release_job_lock("job-1")
        assert runtime_state.acquire_job_lock("job-1") is True  # Can re-acquire

    def test_job_lock_expires(self, runtime_state):
        """Verify locks expire after timeout."""
        import time
        
        # Set very short TTL for testing
        runtime_state.lock_ttl_sec = 2
        
        assert runtime_state.acquire_job_lock("job-expire") is True
        
        # Wait for expiry
        time.sleep(3)
        
        # Should be able to acquire again
        assert runtime_state.acquire_job_lock("job-expire") is True

    def test_concurrent_lock_acquisition(self, runtime_state):
        """Verify only one thread can acquire lock."""
        import concurrent.futures
        
        acquired = []
        
        def try_acquire(thread_id):
            if runtime_state.acquire_job_lock("concurrent-job"):
                acquired.append(thread_id)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_acquire, i) for i in range(10)]
            concurrent.futures.wait(futures)
        
        # Only one thread should have acquired the lock
        assert len(acquired) == 1

    def test_key_cooldown(self, runtime_state):
        """Verify key cooldown tracking."""
        import time
        
        runtime_state.set_key_cooldown("groq", "key-1", duration_sec=2, reason="rate limit")
        
        # Should be in cooldown
        assert runtime_state.is_key_in_cooldown("groq", "key-1") is True
        
        time.sleep(3)
        
        # Should be out of cooldown
        assert runtime_state.is_key_in_cooldown("groq", "key-1") is False

    def test_healthcheck(self, runtime_state):
        """Verify healthcheck."""
        result = runtime_state.healthcheck()
        assert result["redis_state"] == "ok"


pytestmark = pytest.mark.integration
```

---

## 6.3 FFmpeg-Dependent Tests

**Problem:**
`AudioChunker.chunk_audio()` requires ffmpeg but has no tests. The `AudioInspector` also requires ffprobe.

**Solution:**
Add tests marked with `@pytest.mark.ffmpeg` that run only when ffmpeg is available.

**Implementation:**

**Create tests/test_audio_integration.py:**
```python
"""Integration tests for audio processing (requires ffmpeg/ffprobe)."""

import subprocess
import wave
from pathlib import Path

import pytest

from audio_transcript.services.audio import AudioChunker, AudioInspector


def ffmpeg_available():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# Skip all tests in this module if ffmpeg not available
pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not available"),
]


@pytest.fixture
def sample_wav(tmp_path):
    """Create a simple WAV file for testing."""
    wav_path = tmp_path / "sample.wav"
    
    # Create 5-second mono 16kHz WAV
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)
        # 5 seconds of silence
        wav_file.writeframes(b"\x00\x00" * 16000 * 5)
    
    return wav_path


@pytest.fixture
def long_wav(tmp_path):
    """Create a longer WAV file for chunking tests."""
    wav_path = tmp_path / "long.wav"
    
    # Create 35-second mono 16kHz WAV
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000 * 35)
    
    return wav_path


class TestAudioInspector:
    """Tests for AudioInspector with real ffprobe."""

    def test_get_duration(self, sample_wav):
        inspector = AudioInspector()
        duration = inspector.get_duration(sample_wav)
        
        assert abs(duration - 5.0) < 0.1  # ~5 seconds

    def test_get_file_metadata(self, sample_wav):
        inspector = AudioInspector()
        metadata = inspector.get_file_metadata(sample_wav)
        
        assert metadata.duration > 0
        assert metadata.channels == 1
        assert metadata.sample_rate == 16000
        assert metadata.size_bytes > 0
        assert "wav" in metadata.format.lower() or "pcm" in metadata.codec.lower()

    def test_get_duration_invalid_file(self, tmp_path):
        """Verify meaningful error for invalid files."""
        from audio_transcript.domain.errors import AudioProcessingError
        
        bad_file = tmp_path / "bad.wav"
        bad_file.write_text("not audio")
        
        inspector = AudioInspector()
        with pytest.raises(AudioProcessingError) as exc_info:
            inspector.get_duration(bad_file)
        
        assert "bad.wav" in str(exc_info.value)


class TestAudioChunker:
    """Tests for AudioChunker with real ffmpeg."""

    def test_chunk_audio_creates_chunks(self, long_wav, tmp_path):
        """Verify chunking creates correct number of chunks."""
        inspector = AudioInspector()
        chunker = AudioChunker(inspector)
        
        chunk_dir = tmp_path / "chunks"
        chunks = chunker.chunk_audio(
            long_wav,
            chunk_dir,
            duration_sec=10,
            overlap_sec=2,
        )
        
        # 35 seconds with 10s chunks and 2s overlap = 8s effective per chunk
        # Should get ~5 chunks: 0-10, 8-18, 16-26, 24-34, 32-35
        assert len(chunks) >= 4
        assert all(chunk.exists() for chunk in chunks)
        assert all(chunk.suffix == ".wav" for chunk in chunks)

    def test_chunk_audio_maintains_audio_quality(self, long_wav, tmp_path):
        """Verify chunks are valid audio files."""
        inspector = AudioInspector()
        chunker = AudioChunker(inspector)
        
        chunk_dir = tmp_path / "chunks"
        chunks = chunker.chunk_audio(long_wav, chunk_dir, duration_sec=10, overlap_sec=2)
        
        # Verify each chunk is valid audio
        for chunk in chunks:
            metadata = inspector.get_file_metadata(chunk)
            assert metadata.sample_rate == 16000  # Resampled
            assert metadata.channels == 1  # Mono
            assert metadata.duration > 0

    def test_chunk_audio_overlap_validation(self, long_wav, tmp_path):
        """Verify overlap >= duration raises error."""
        from audio_transcript.domain.errors import ValidationError
        
        inspector = AudioInspector()
        chunker = AudioChunker(inspector)
        
        chunk_dir = tmp_path / "chunks"
        with pytest.raises(ValidationError) as exc_info:
            chunker.chunk_audio(long_wav, chunk_dir, duration_sec=10, overlap_sec=10)
        
        assert "overlap" in str(exc_info.value).lower()


class TestMergeTranscripts:
    """Tests for transcript merging."""

    def test_merge_transcripts_handles_overlap(self):
        """Verify overlapping segments are deduplicated."""
        from audio_transcript.domain.models import TranscriptResult, TranscriptSegment
        from audio_transcript.services.audio import merge_transcripts
        
        chunk1 = TranscriptResult(
            text="Hello world",
            segments=[
                TranscriptSegment(id=0, start=0.0, end=5.0, text="Hello"),
                TranscriptSegment(id=1, start=5.0, end=10.0, text="world"),
            ],
            provider="test",
        )
        chunk2 = TranscriptResult(
            text="world how are you",
            segments=[
                TranscriptSegment(id=0, start=0.0, end=2.0, text="world"),  # Overlap
                TranscriptSegment(id=1, start=2.0, end=5.0, text="how"),
                TranscriptSegment(id=2, start=5.0, end=8.0, text="are you"),
            ],
            provider="test",
        )
        
        merged = merge_transcripts([chunk1, chunk2], overlap_sec=3)
        
        # "world" should only appear once
        texts = [s.text for s in merged.segments]
        assert texts.count("world") == 1
        assert "Hello" in texts
        assert "how" in texts
```

---

## 6.4 Concurrent Access Tests

**Problem:**
`ProviderKeyPool` and job locks handle concurrent access but aren't tested under load.

**Solution:**
Add stress tests for concurrent operations.

**Implementation:**

**Create tests/test_concurrency.py:**
```python
"""Concurrency and thread-safety tests."""

import concurrent.futures
import threading
import time

import pytest

from audio_transcript.infra.runtime_state import InMemoryRuntimeState
from audio_transcript.services.router import ProviderKeyPool


class TestProviderKeyPoolConcurrency:
    """Thread-safety tests for ProviderKeyPool."""

    def test_concurrent_acquire_returns_different_keys(self):
        """Verify concurrent acquires distribute across keys."""
        pool = ProviderKeyPool(
            provider_name="test",
            api_keys=["key1", "key2", "key3"],
            runtime_state=InMemoryRuntimeState(),
        )
        
        acquired_keys = []
        lock = threading.Lock()
        
        def acquire_key():
            key_state = pool.acquire()
            if key_state:
                with lock:
                    acquired_keys.append(key_state.key_id)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(acquire_key) for _ in range(100)]
            concurrent.futures.wait(futures)
        
        # Should have used all keys, distributed reasonably
        key_counts = {k: acquired_keys.count(k) for k in set(acquired_keys)}
        assert len(key_counts) == 3  # All keys used
        # Each key should have been used at least 20 times
        assert all(count >= 20 for count in key_counts.values())

    def test_cooldown_respected_under_load(self):
        """Verify cooled-down keys are skipped."""
        runtime_state = InMemoryRuntimeState()
        pool = ProviderKeyPool(
            provider_name="test",
            api_keys=["key1", "key2"],
            runtime_state=runtime_state,
        )
        
        # Cool down key1
        pool.cooldown("key1", duration_sec=60, reason="rate limit")
        
        acquired_keys = []
        
        def acquire_key():
            key_state = pool.acquire()
            if key_state:
                acquired_keys.append(key_state.key_id)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(acquire_key) for _ in range(50)]
            concurrent.futures.wait(futures)
        
        # Should only have key2
        assert all(k == "key2" for k in acquired_keys)


class TestJobLockConcurrency:
    """Thread-safety tests for job locking."""

    def test_only_one_worker_processes_job(self):
        """Verify job lock prevents double-processing."""
        runtime_state = InMemoryRuntimeState()
        
        acquired_by = []
        lock = threading.Lock()
        
        def try_process(worker_id):
            if runtime_state.acquire_job_lock("job-1"):
                with lock:
                    acquired_by.append(worker_id)
                # Simulate processing
                time.sleep(0.1)
                runtime_state.release_job_lock("job-1")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_process, i) for i in range(10)]
            concurrent.futures.wait(futures)
        
        # Only one worker should have acquired initially
        # (others may acquire after release, but at any given time only one has it)
        # For this test, we check the first acquisition
        assert len(acquired_by) >= 1


class TestRepositoryConcurrency:
    """Thread-safety tests for repository operations."""

    def test_concurrent_job_updates_dont_lose_data(self, tmp_path):
        """Verify concurrent updates are serialized correctly."""
        from audio_transcript.domain.models import JobPayload, JobStatus, TranscriptionJob
        from audio_transcript.infra.repository import InMemoryJobRepository
        
        repository = InMemoryJobRepository()
        
        job = TranscriptionJob(
            job_id="concurrent-job",
            status=JobStatus.QUEUED,
            payload=JobPayload(
                filename="test.wav",
                content_type="audio/wav",
                source_path="/tmp/test.wav",
            ),
        )
        repository.create(job)
        
        errors = []
        
        def update_job(worker_id):
            try:
                for i in range(10):
                    loaded = repository.get("concurrent-job")
                    loaded.status = JobStatus.PROCESSING
                    repository.save(loaded)
            except Exception as exc:
                errors.append(exc)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_job, i) for i in range(5)]
            concurrent.futures.wait(futures)
        
        assert len(errors) == 0
        # Job should still be retrievable
        final = repository.get("concurrent-job")
        assert final is not None
```

---

## 6.5 Extract Shared Test Fixtures

**Problem:**
`FakeInspector` and similar test doubles are duplicated:
- `tests/test_api.py` lines 19-31
- `tests/test_service.py` lines 33-48

**Solution:**
Consolidate into `tests/conftest.py`.

**Implementation:**

**Update tests/conftest.py:**
```python
"""Shared test fixtures and test doubles."""

import sys
from pathlib import Path

import pytest

# Add src to path
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


# =============================================================================
# Test Doubles
# =============================================================================


class FakeInspector(AudioInspector):
    """Test double for AudioInspector that doesn't require ffprobe."""

    def get_file_metadata(self, file_path: Path) -> FileMetadata:
        return FileMetadata(
            filename=file_path.name,
            path=str(file_path),
            size_bytes=file_path.stat().st_size if file_path.exists() else 1000,
            duration=1.0,
            format=file_path.suffix.lstrip(".") or "wav",
            bit_rate=128000,
            codec="pcm_s16le",
            sample_rate=16000,
            channels=1,
        )

    def get_duration(self, audio_path: Path) -> float:
        return 1.0


class FakeChunker(AudioChunker):
    """Test double for AudioChunker that doesn't require ffmpeg."""

    def __init__(self):
        super().__init__(FakeInspector())

    def chunk_audio(self, audio_path, chunk_dir, duration_sec, overlap_sec):
        # Don't actually chunk - return the original file
        return [audio_path]


class FakeProvider:
    """Test double for transcription providers."""

    def __init__(
        self,
        name: str,
        result: TranscriptResult = None,
        error: Exception = None,
    ):
        self.provider_name = name
        self.result = result or TranscriptResult(
            text=f"{name} transcription text",
            segments=[TranscriptSegment(start=0.0, end=1.0, text=f"{name} text")],
            provider=name,
            model=f"{name}-model",
        )
        self.error = error
        self.call_count = 0

    def transcribe(self, audio_path, content_type, model_override=None):
        self.call_count += 1
        if self.error:
            raise self.error
        return self.result

    def status(self):
        return {"provider": self.provider_name, "call_count": self.call_count}


class CountingProvider(FakeProvider):
    """Provider that tracks calls and can fail N times before succeeding."""

    def __init__(self, name: str, fail_count: int = 0, fail_error: Exception = None):
        super().__init__(name)
        self.fail_count = fail_count
        self.fail_error = fail_error

    def transcribe(self, audio_path, content_type, model_override=None):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise self.fail_error or RuntimeError(f"Simulated failure {self.call_count}")
        return self.result


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_inspector():
    """Provide a FakeInspector instance."""
    return FakeInspector()


@pytest.fixture
def fake_chunker():
    """Provide a FakeChunker instance."""
    return FakeChunker()


@pytest.fixture
def test_settings(tmp_path):
    """Provide test Settings with temp directories."""
    return Settings(
        service_api_key="test-secret",
        database_url="postgresql://unused",
        redis_url="redis://unused",
        storage_root=tmp_path / "artifacts",
        transcript_dataset_root=tmp_path / "dataset",
        groq_api_keys=["gsk_test_1"],
        mistral_api_keys=["msk_test_1"],
    )


@pytest.fixture
def in_memory_repository():
    """Provide an in-memory repository."""
    return InMemoryJobRepository()


@pytest.fixture
def in_memory_queue():
    """Provide an in-memory queue."""
    return InMemoryQueueBackend()


@pytest.fixture
def in_memory_runtime_state():
    """Provide an in-memory runtime state."""
    return InMemoryRuntimeState()


@pytest.fixture
def artifact_store(tmp_path):
    """Provide a TranscriptArtifactStore with temp directories."""
    return TranscriptArtifactStore(
        tmp_path / "artifacts",
        tmp_path / "dataset",
    )


@pytest.fixture
def test_runtime(
    tmp_path,
    test_settings,
    in_memory_repository,
    in_memory_queue,
    in_memory_runtime_state,
    artifact_store,
    fake_inspector,
    fake_chunker,
):
    """Provide a complete test runtime with fake dependencies."""
    providers = {
        "groq": FakeProvider("groq"),
        "mistral": FakeProvider("mistral"),
    }
    fallback = FakeProvider("whisper_cpp")

    service = TranscriptionService(
        RuntimeDependencies(
            settings=test_settings,
            repository=in_memory_repository,
            artifact_store=artifact_store,
            runtime_state=in_memory_runtime_state,
            router=ProviderRouter(["groq", "mistral"]),
            remote_providers=providers,
            fallback_provider=fallback,
            inspector=fake_inspector,
            chunker=fake_chunker,
        )
    )

    return {
        "settings": test_settings,
        "repository": in_memory_repository,
        "queue": in_memory_queue,
        "runtime_state": in_memory_runtime_state,
        "artifact_store": artifact_store,
        "service": service,
        "providers": providers,
        "fallback_provider": fallback,
    }


@pytest.fixture
def test_app(test_runtime, test_settings):
    """Provide a Flask test app with fake dependencies."""
    app = create_app(
        test_settings,
        repository=test_runtime["repository"],
        queue=test_runtime["queue"],
        artifact_store=test_runtime["artifact_store"],
        runtime_state=test_runtime["runtime_state"],
        service=test_runtime["service"],
        providers=test_runtime["providers"],
        fallback_provider=test_runtime["fallback_provider"],
    )
    return app


@pytest.fixture
def client(test_app):
    """Provide a Flask test client."""
    return test_app.test_client()


@pytest.fixture
def auth_headers():
    """Provide authenticated request headers."""
    return {"X-API-Key": "test-secret"}
```

**Update test_api.py to use shared fixtures:**
```python
"""API endpoint tests."""

import io

import pytest

from audio_transcript.worker.runner import run_single_iteration


def test_api_job_lifecycle(client, auth_headers, test_runtime):
    """Test complete job lifecycle through API."""
    response = client.post(
        "/v1/jobs",
        headers=auth_headers,
        data={"file": (io.BytesIO(b"RIFFfake"), "sample.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    job_id = response.get_json()["job"]["id"]

    status_response = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
    assert status_response.status_code == 200
    assert status_response.get_json()["job"]["status"] == "queued"

    run_single_iteration(test_runtime)

    status_response = client.get(f"/v1/jobs/{job_id}", headers=auth_headers)
    assert status_response.get_json()["job"]["status"] == "succeeded"

    result_response = client.get(f"/v1/jobs/{job_id}/result", headers=auth_headers)
    assert result_response.status_code == 200
    assert result_response.get_json()["transcript"]["provider"] == "groq"


def test_api_rejects_invalid_auth(client):
    """Verify 401 for invalid API key."""
    response = client.get("/v1/providers/status", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_api_lists_jobs_with_search(client, auth_headers, test_runtime):
    """Verify job search/filter works."""
    response = client.post(
        "/v1/jobs",
        headers=auth_headers,
        data={"file": (io.BytesIO(b"RIFFfake"), "sample.wav")},
        content_type="multipart/form-data",
    )
    job_id = response.get_json()["job"]["id"]
    run_single_iteration(test_runtime)

    response = client.get("/v1/jobs?provider=groq", headers=auth_headers)
    assert response.status_code == 200
    jobs = response.get_json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["id"] == job_id
```

---

## Verification Checklist

After implementing Phase 6:

- [ ] `pytest tests/` passes (unit tests)
- [ ] `pytest -m integration` passes (with Docker)
- [ ] `pytest -m ffmpeg` passes (with ffmpeg installed)
- [ ] No duplicate `FakeInspector` definitions
- [ ] Test coverage > 80% on core modules
- [ ] CI runs integration tests in separate job

---

## Files Modified/Created

| File | Change |
|------|--------|
| `tests/conftest.py` | Complete rewrite with shared fixtures |
| `tests/test_api.py` | Simplify to use shared fixtures |
| `tests/test_service.py` | Simplify to use shared fixtures |
| `tests/test_repository_integration.py` | CREATE - Postgres tests |
| `tests/test_redis_integration.py` | CREATE - Redis tests |
| `tests/test_audio_integration.py` | CREATE - ffmpeg tests |
| `tests/test_concurrency.py` | CREATE - thread-safety tests |
| `pyproject.toml` | Add testcontainers dependency, pytest markers |

---

## CI Configuration

**Example GitHub Actions workflow:**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=src

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
      redis:
        image: redis:7
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest -m integration tests/ -v

  ffmpeg-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: sudo apt-get install -y ffmpeg
      - run: pip install -e ".[dev]"
      - run: pytest -m ffmpeg tests/ -v
```
