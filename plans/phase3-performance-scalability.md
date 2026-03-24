# Phase 3: Performance & Scalability

Priority: MEDIUM | Effort: MEDIUM-HIGH | Risk if Skipped: Performance bottlenecks at scale

This phase addresses performance issues that will become problematic under load.

---

## 3.1 Optimize Job Attempts Persistence

**File:** `src/audio_transcript/infra/repository.py`

**Problem:**
Lines 127-128 delete and reinsert all attempts on every save:
```python
cur.execute("DELETE FROM job_attempts WHERE job_id = %s", (job.job_id,))
for attempt in job.attempts:
    cur.execute("INSERT INTO job_attempts ...")
```

This causes:
- Unnecessary I/O on every job update
- Auto-increment ID gaps
- Potential index churn
- Slower writes as attempts accumulate

**Solution:**
Use upsert pattern with attempt tracking.

**Implementation:**

**Option A: Hash-based deduplication (simpler)**
```python
def save(self, job: TranscriptionJob) -> None:
    # ... existing job upsert logic ...
    
    with self._get_connection() as conn, conn.cursor() as cur:
        # ... job INSERT ON CONFLICT ...
        
        # Only insert new attempts (compare by composite key)
        if job.attempts:
            # Get existing attempt count
            cur.execute(
                "SELECT COUNT(*) FROM job_attempts WHERE job_id = %s",
                (job.job_id,)
            )
            existing_count = cur.fetchone()[0]
            
            # Only insert new attempts
            new_attempts = job.attempts[existing_count:]
            for attempt in new_attempts:
                cur.execute(
                    """
                    INSERT INTO job_attempts (
                        job_id, provider, key_id_masked, success, retryable, error, status_code,
                        model, latency_ms, started_at, finished_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job.job_id,
                        attempt.provider,
                        attempt.key_id,
                        attempt.success,
                        attempt.retryable,
                        attempt.error,
                        attempt.status_code,
                        attempt.model,
                        attempt.latency_ms,
                        attempt.started_at,
                        attempt.finished_at,
                    ),
                )
```

**Option B: Upsert with unique constraint (more robust)**

Add unique constraint on attempts:
```sql
ALTER TABLE job_attempts ADD CONSTRAINT uq_job_attempts_started 
    UNIQUE (job_id, provider, started_at);
```

Then use upsert:
```python
def save(self, job: TranscriptionJob) -> None:
    with self._get_connection() as conn, conn.cursor() as cur:
        # ... job upsert ...
        
        for attempt in job.attempts:
            cur.execute(
                """
                INSERT INTO job_attempts (
                    job_id, provider, key_id_masked, success, retryable, error, status_code,
                    model, latency_ms, started_at, finished_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id, provider, started_at) DO UPDATE SET
                    success = EXCLUDED.success,
                    retryable = EXCLUDED.retryable,
                    error = EXCLUDED.error,
                    status_code = EXCLUDED.status_code,
                    model = EXCLUDED.model,
                    latency_ms = EXCLUDED.latency_ms,
                    finished_at = EXCLUDED.finished_at
                """,
                (
                    job.job_id,
                    attempt.provider,
                    attempt.key_id,
                    attempt.success,
                    attempt.retryable,
                    attempt.error,
                    attempt.status_code,
                    attempt.model,
                    attempt.latency_ms,
                    attempt.started_at,
                    attempt.finished_at,
                ),
            )
```

**Migration:**
```sql
-- Add unique constraint for upsert support
ALTER TABLE job_attempts 
ADD CONSTRAINT uq_job_attempts_job_provider_started 
UNIQUE (job_id, provider, started_at);
```

**Benchmark:**
| Scenario | Before | After |
|----------|--------|-------|
| Job with 1 attempt | 2 queries | 1 query |
| Job with 5 attempts | 6 queries | 1 query (upsert all) |
| 100 jobs/min | ~600 queries/min | ~100 queries/min |

---

## 3.2 Parallel Chunk Transcription

**File:** `src/audio_transcript/services/transcription.py`

**Problem:**
Lines 102-105 process chunks sequentially:
```python
chunk_results = [
    self._transcribe_single(chunk_path, "audio/wav", job.payload.model_override, job)
    for chunk_path in chunk_paths
]
```

For a 1-hour audio file with 10-minute chunks, this means 6 sequential API calls.

**Solution:**
Use `concurrent.futures.ThreadPoolExecutor` for parallel chunk processing.

**Implementation:**
```python
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
# ... other imports


@dataclass
class RuntimeDependencies:
    """Shared runtime dependencies."""
    # ... existing fields ...
    max_parallel_chunks: int = 3  # New: configurable parallelism


class TranscriptionService:
    """Coordinates job execution."""

    def __init__(self, deps: RuntimeDependencies):
        self.deps = deps
        self.logger = logging.getLogger("audio_transcript.transcription")
        self._chunk_executor = ThreadPoolExecutor(
            max_workers=deps.max_parallel_chunks,
            thread_name_prefix="chunk-transcribe-"
        )

    def _transcribe(self, audio_path: Path, job: TranscriptionJob) -> TranscriptResult:
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        chunk_duration = job.payload.chunk_duration_sec or self.deps.settings.chunk_duration_sec
        chunk_overlap = job.payload.chunk_overlap_sec or self.deps.settings.chunk_overlap_sec

        if file_size_mb <= self.deps.settings.max_file_size_mb:
            return self._transcribe_single(audio_path, job.payload.content_type, job.payload.model_override, job)

        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_paths = self.deps.chunker.chunk_audio(audio_path, Path(temp_dir), chunk_duration, chunk_overlap)
            chunk_results = self._transcribe_chunks_parallel(chunk_paths, job)
        
        return merge_transcripts(chunk_results, chunk_overlap)

    def _transcribe_chunks_parallel(
        self,
        chunk_paths: List[Path],
        job: TranscriptionJob,
    ) -> List[TranscriptResult]:
        """Transcribe multiple chunks in parallel, maintaining order."""
        results: Dict[int, TranscriptResult] = {}
        errors: List[Tuple[int, Exception]] = []

        # Submit all chunks
        future_to_index = {
            self._chunk_executor.submit(
                self._transcribe_single,
                chunk_path,
                "audio/wav",
                job.payload.model_override,
                job,
            ): index
            for index, chunk_path in enumerate(chunk_paths)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
                self.logger.info(
                    "Chunk %d/%d completed for job %s",
                    index + 1, len(chunk_paths), job.job_id
                )
            except Exception as exc:
                errors.append((index, exc))
                self.logger.error(
                    "Chunk %d/%d failed for job %s: %s",
                    index + 1, len(chunk_paths), job.job_id, exc
                )

        # If any chunk failed, raise the first error
        if errors:
            errors.sort(key=lambda x: x[0])  # Sort by chunk index
            raise errors[0][1]

        # Return results in original order
        return [results[i] for i in range(len(chunk_paths))]

    def shutdown(self):
        """Gracefully shutdown the executor."""
        self._chunk_executor.shutdown(wait=True)
```

**Configuration:**
Add to `config.py`:
```python
@dataclass
class Settings:
    # ... existing fields ...
    max_parallel_chunks: int = 3  # Concurrent chunk transcriptions
```

And environment parsing:
```python
max_parallel_chunks=_parse_int("MAX_PARALLEL_CHUNKS", 3),
```

**Considerations:**
- Set `max_parallel_chunks` based on provider rate limits
- Groq has per-minute request limits, so 3 parallel is conservative
- Consider separate limits per provider
- Thread-safety: each chunk creates independent provider calls

**Benchmark:**
| Scenario | Sequential | Parallel (3) |
|----------|------------|--------------|
| 6 chunks @ 30s each | 180s | ~60s |
| 12 chunks @ 30s each | 360s | ~120s |

**Test:**
```python
def test_parallel_chunk_transcription(tmp_path):
    """Verify chunks are processed in parallel."""
    import time
    
    call_times = []
    
    class TimingProvider:
        provider_name = "timing"
        def transcribe(self, *args, **kwargs):
            call_times.append(time.time())
            time.sleep(0.1)  # Simulate API latency
            return TranscriptResult(text="chunk", segments=[], provider="timing")
    
    # Setup with 6 chunks
    # ... create service with max_parallel_chunks=3 ...
    
    start = time.time()
    # Process job with 6 chunks
    elapsed = time.time() - start
    
    # With 3 parallel workers, 6 chunks should complete in ~2 batches
    assert elapsed < 0.4  # Should be ~0.2s, not 0.6s sequential
```

---

## 3.3 Streaming Parquet Reads

**File:** `src/audio_transcript/infra/storage.py`

**Problem:**
Lines 89-92 load entire Parquet file into memory:
```python
segments = pq.ParquetFile(segments_path).read().to_pylist()
```

For large transcripts (e.g., 10-hour podcast with 3000+ segments), this can:
- Cause memory spikes
- Slow response times
- Risk OOM on constrained systems

**Solution:**
Use row group iteration for streaming reads, with optional pagination.

**Implementation:**
```python
"""Artifact storage for uploads and transcript Parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ..domain.errors import ArtifactNotFoundError, ValidationError
from ..domain.models import FileMetadata, TranscriptResult, TranscriptionJob


class TranscriptArtifactStore:
    """Manage uploaded files and Parquet-backed transcript artifacts."""

    # ... existing __init__, job_dir, save_upload, _partition_dir, save_result ...

    def load_result(
        self,
        artifact_uri: str | Path,
        segment_offset: int = 0,
        segment_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load transcript result with optional segment pagination.
        
        Args:
            artifact_uri: Path to artifact directory or JSON file
            segment_offset: Start index for segments (0-based)
            segment_limit: Maximum segments to return (None = all)
        
        Returns:
            Result dict with job, file_metadata, transcript, attempts
        """
        artifact_dir = Path(artifact_uri)
        if artifact_dir.is_file() and artifact_dir.suffix.lower() == ".json":
            with open(artifact_dir, "r", encoding="utf-8") as handle:
                return json.load(handle)

        segments_path = artifact_dir / "segments.parquet"
        summary_path = artifact_dir / "summary.parquet"
        if not segments_path.exists() or not summary_path.exists():
            raise ArtifactNotFoundError(f"Missing Parquet artifact under {artifact_dir}")

        # Load summary (always small)
        summary_rows = pq.ParquetFile(summary_path).read().to_pylist()
        if not summary_rows:
            raise ArtifactNotFoundError(f"Empty summary artifact under {artifact_dir}")
        summary = summary_rows[0]

        # Stream segments with pagination
        segments = list(self._iter_segments(
            segments_path,
            offset=segment_offset,
            limit=segment_limit,
        ))

        # Get total segment count for pagination info
        total_segments = summary.get("segment_count", len(segments))

        result = {
            "job": {
                "id": summary["job_id"],
                "status": summary["status"],
                "filename": summary["source_filename"],
                "created_at": summary["created_at"],
                "started_at": summary["started_at"],
                "completed_at": summary["completed_at"],
            },
            "file_metadata": {
                "filename": summary["file_metadata_filename"],
                "path": summary["file_metadata_path"],
                "size_bytes": summary["size_bytes"],
                "duration": summary["duration_sec"],
                "format": summary["format"],
                "bit_rate": summary["bit_rate"],
                "codec": summary["codec"],
                "sample_rate": summary["sample_rate"],
                "channels": summary["channels"],
            },
            "transcript": {
                "text": summary["text"],
                "provider": summary["provider"],
                "model": summary["model"],
                "segments": segments,
                "total_segments": total_segments,
            },
            "attempts": json.loads(summary["attempts_json"]),
        }

        # Add pagination metadata if paginated
        if segment_offset > 0 or segment_limit is not None:
            result["pagination"] = {
                "offset": segment_offset,
                "limit": segment_limit,
                "total": total_segments,
                "has_more": segment_offset + len(segments) < total_segments,
            }

        return result

    def _iter_segments(
        self,
        segments_path: Path,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Iterate segments from Parquet file with streaming.
        
        Uses row group iteration to avoid loading entire file into memory.
        """
        pf = pq.ParquetFile(segments_path)
        current_index = 0
        yielded = 0
        effective_limit = limit if limit is not None else float('inf')

        for batch in pf.iter_batches(batch_size=1000):
            rows = batch.to_pylist()
            for row in rows:
                if current_index < offset:
                    current_index += 1
                    continue
                
                if yielded >= effective_limit:
                    return
                
                segment = self._segment_to_result_item(current_index, row)
                if segment:  # Skip empty placeholder rows
                    yield segment
                    yielded += 1
                
                current_index += 1

    def iter_all_segments(
        self,
        artifact_uri: str | Path,
        batch_size: int = 1000,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream all segments without loading into memory.
        
        Useful for export or processing pipelines.
        """
        segments_path = Path(artifact_uri) / "segments.parquet"
        if not segments_path.exists():
            raise ArtifactNotFoundError(f"Missing segments at {segments_path}")
        
        yield from self._iter_segments(segments_path)

    # ... existing _segment_rows, _summary_row, _segment_to_result_item ...
```

**API Endpoint Update:**
Add pagination to routes.py:
```python
@bp.get("/jobs/<job_id>/result")
@require_api_key
def get_result(job_id: str):
    job = current_app.config["repository"].get(job_id)
    if job.status != JobStatus.SUCCEEDED or not job.result_path:
        return (
            jsonify({"error": {"code": "result_unavailable", "message": "Result is not ready", "details": {}}}),
            409,
        )
    
    # Support optional pagination
    offset = int(request.args.get("segment_offset", 0))
    limit = request.args.get("segment_limit")
    limit = int(limit) if limit else None
    
    document = current_app.config["artifact_store"].load_result(
        job.result_path,
        segment_offset=offset,
        segment_limit=limit,
    )
    return jsonify(document)
```

**Memory Comparison:**
| Segments | Before (load all) | After (streaming) |
|----------|-------------------|-------------------|
| 100 | ~50KB | ~50KB |
| 1,000 | ~500KB | ~50KB per batch |
| 10,000 | ~5MB | ~50KB per batch |

**Test:**
```python
def test_load_result_with_pagination(tmp_path):
    """Verify paginated segment loading."""
    store = TranscriptArtifactStore(tmp_path / "art", tmp_path / "ds")
    
    # Create job with many segments
    # ... setup with 100 segments ...
    
    # Load first page
    result = store.load_result(artifact_path, segment_offset=0, segment_limit=10)
    assert len(result["transcript"]["segments"]) == 10
    assert result["pagination"]["has_more"] is True
    assert result["pagination"]["total"] == 100
    
    # Load second page
    result = store.load_result(artifact_path, segment_offset=10, segment_limit=10)
    assert len(result["transcript"]["segments"]) == 10
    assert result["pagination"]["offset"] == 10
```

---

## Verification Checklist

After implementing Phase 3:

- [ ] `pytest tests/` passes
- [ ] Benchmark job save with 5 attempts: verify single query
- [ ] Benchmark 12-chunk file: verify ~3x speedup with parallel=3
- [ ] Load 10,000-segment transcript: verify memory stays bounded
- [ ] API pagination works: `/v1/jobs/{id}/result?segment_offset=0&segment_limit=100`
- [ ] No race conditions in parallel chunk transcription
- [ ] Graceful shutdown: verify executor cleanup

---

## Files Modified

| File | Change |
|------|--------|
| `src/audio_transcript/infra/repository.py` | Upsert for attempts |
| `src/audio_transcript/services/transcription.py` | Parallel chunk processing |
| `src/audio_transcript/infra/storage.py` | Streaming Parquet reads, pagination |
| `src/audio_transcript/api/routes.py` | Pagination query params |
| `src/audio_transcript/config.py` | Add max_parallel_chunks setting |

---

## Configuration Changes

Add to `.env.example`:
```bash
# Performance tuning
MAX_PARALLEL_CHUNKS=3  # Concurrent chunk transcriptions (default: 3)
```

---

## Database Migration

```sql
-- Migration: Add unique constraint for attempt upserts
-- File: migrations/003_attempt_upsert.sql

ALTER TABLE job_attempts 
ADD CONSTRAINT uq_job_attempts_job_provider_started 
UNIQUE (job_id, provider, started_at);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_job_attempts_lookup 
ON job_attempts (job_id, provider, started_at);
```
