# Phase 4: Error Handling & Robustness

Priority: MEDIUM | Effort: LOW-MEDIUM | Risk if Skipped: Silent failures, data corruption

This phase improves error handling to provide clear failure messages and prevent partial/corrupt state.

---

## 4.1 Wrap ffmpeg/ffprobe Errors

**File:** `src/audio_transcript/services/audio.py`

**Problem:**
Lines 28 and 63 use `subprocess.run(..., check=True)` without wrapping errors:
```python
result = subprocess.run(cmd, capture_output=True, text=True, check=True)
```

When ffprobe/ffmpeg fails, users see generic `CalledProcessError` with subprocess details instead of user-friendly messages.

**Current behavior:**
```
subprocess.CalledProcessError: Command '['ffprobe', ...]' returned non-zero exit status 1.
```

**Desired behavior:**
```
AudioProcessingError: Failed to read audio metadata: ffprobe error - Invalid data found when processing input
```

**Solution:**
Catch subprocess errors and convert to domain `AudioProcessingError`.

**Implementation:**

**Add to domain/errors.py:**
```python
class AudioProcessingError(AudioTranscriptError):
    """Raised when audio file processing fails (ffmpeg/ffprobe errors)."""
```

**Update audio.py:**
```python
"""Audio inspection and chunking helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List

from ..domain.errors import AudioProcessingError, ValidationError
from ..domain.models import FileMetadata, TranscriptResult, TranscriptSegment


class AudioInspector:
    """Audio metadata utilities backed by ffprobe."""

    def get_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
            
        Raises:
            AudioProcessingError: If ffprobe fails to read the file
        """
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "Unknown error"
            raise AudioProcessingError(
                f"Failed to read audio duration from '{audio_path.name}': {stderr}"
            ) from exc
        except ValueError as exc:
            raise AudioProcessingError(
                f"Invalid duration value from '{audio_path.name}'"
            ) from exc

    def get_file_metadata(self, file_path: Path) -> FileMetadata:
        """Extract comprehensive audio metadata.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            FileMetadata with format, codec, duration, etc.
            
        Raises:
            AudioProcessingError: If ffprobe fails
        """
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            raw_metadata = json.loads(result.stdout)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "Unknown error"
            raise AudioProcessingError(
                f"Failed to read audio metadata from '{file_path.name}': {stderr}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise AudioProcessingError(
                f"Invalid metadata format from '{file_path.name}': {exc}"
            ) from exc

        format_info = raw_metadata.get("format", {})
        stream_info = raw_metadata.get("streams", [{}])[0] if raw_metadata.get("streams") else {}

        # Validate required fields
        if not format_info:
            raise AudioProcessingError(
                f"No format information found in '{file_path.name}' - file may be corrupted or not an audio file"
            )

        return FileMetadata(
            filename=format_info.get("filename", ""),
            path=str(file_path),
            size_bytes=int(format_info.get("size", 0)),
            duration=float(format_info.get("duration", 0)),
            format=format_info.get("format_name", ""),
            bit_rate=int(format_info.get("bit_rate", 0)),
            codec=stream_info.get("codec_name", ""),
            sample_rate=int(stream_info.get("sample_rate", 0) or 0),
            channels=int(stream_info.get("channels", 0) or 0),
        )


class AudioChunker:
    """Split long audio files into overlapping wav chunks."""

    def __init__(self, inspector: AudioInspector):
        self.inspector = inspector

    def chunk_audio(
        self,
        audio_path: Path,
        chunk_dir: Path,
        duration_sec: int,
        overlap_sec: int,
    ) -> List[Path]:
        """Split audio into overlapping chunks.
        
        Args:
            audio_path: Source audio file
            chunk_dir: Directory for chunk outputs
            duration_sec: Duration of each chunk
            overlap_sec: Overlap between adjacent chunks
            
        Returns:
            List of paths to chunk files
            
        Raises:
            ValidationError: If duration <= overlap
            AudioProcessingError: If ffmpeg fails to process
        """
        if duration_sec <= overlap_sec:
            raise ValidationError("chunk duration must be greater than overlap")

        try:
            total_duration = self.inspector.get_duration(audio_path)
        except AudioProcessingError:
            raise  # Re-raise with context already set

        chunk_paths = []
        start_times = []
        current_start = 0.0
        while current_start < total_duration:
            start_times.append(current_start)
            current_start += duration_sec - overlap_sec

        chunk_dir.mkdir(parents=True, exist_ok=True)
        for index, start_time in enumerate(start_times):
            end_time = min(start_time + duration_sec, total_duration)
            chunk_path = chunk_dir / f"chunk_{index:03d}.wav"
            
            try:
                self._create_chunk(audio_path, chunk_path, start_time, end_time)
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode() if exc.stderr else "Unknown error"
                raise AudioProcessingError(
                    f"Failed to create chunk {index} from '{audio_path.name}': {stderr}"
                ) from exc
            
            chunk_paths.append(chunk_path)
        return chunk_paths

    def _create_chunk(
        self,
        audio_path: Path,
        chunk_path: Path,
        start_time: float,
        end_time: float,
    ) -> None:
        """Create a single audio chunk with ffmpeg.
        
        Raises:
            subprocess.CalledProcessError: If ffmpeg fails
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(chunk_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
```

**Error Handler Update (api/errors.py):**
```python
from ..domain.errors import (
    AudioTranscriptError,
    AudioProcessingError,
    AuthenticationError,
    JobNotFoundError,
    ValidationError,
)

def register_error_handlers(app) -> None:
    # ... existing handlers ...

    @app.errorhandler(AudioProcessingError)
    def handle_audio_processing(exc):
        return jsonify({
            "error": {
                "code": "audio_processing_error",
                "message": str(exc),
                "details": {}
            }
        }), 422  # Unprocessable Entity
```

**Test:**
```python
def test_audio_inspector_handles_invalid_file(tmp_path):
    """Verify meaningful error for corrupt audio."""
    bad_file = tmp_path / "bad.wav"
    bad_file.write_bytes(b"not audio data")
    
    inspector = AudioInspector()
    with pytest.raises(AudioProcessingError) as exc_info:
        inspector.get_file_metadata(bad_file)
    
    assert "bad.wav" in str(exc_info.value)
    assert "Failed to read audio metadata" in str(exc_info.value)


def test_audio_chunker_handles_ffmpeg_failure(tmp_path, mocker):
    """Verify ffmpeg errors are wrapped."""
    mocker.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Conversion failed"))
    
    inspector = AudioInspector()
    chunker = AudioChunker(inspector)
    
    with pytest.raises(AudioProcessingError) as exc_info:
        chunker._create_chunk(Path("test.wav"), Path("out.wav"), 0, 10)
    
    assert "Conversion failed" in str(exc_info.value)
```

---

## 4.2 Storage Atomic Writes

**File:** `src/audio_transcript/infra/storage.py`

**Problem:**
Lines 55-56 write Parquet files directly:
```python
pq.write_table(pa.Table.from_pylist(segment_rows), segments_path)
pq.write_table(pa.Table.from_pylist([summary_row]), summary_path)
```

If the process crashes mid-write:
- Partial/corrupt Parquet files
- segments.parquet written but not summary.parquet
- Inconsistent state that breaks `load_result`

**Solution:**
Write to temp files first, then atomic rename.

**Implementation:**
```python
"""Artifact storage for uploads and transcript Parquet datasets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ..domain.errors import ArtifactNotFoundError, StorageError, ValidationError
from ..domain.models import FileMetadata, TranscriptResult, TranscriptionJob


class TranscriptArtifactStore:
    """Manage uploaded files and Parquet-backed transcript artifacts."""

    def __init__(self, root: Path, dataset_root: Path):
        self.root = root
        self.dataset_root = dataset_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_root.mkdir(parents=True, exist_ok=True)

    # ... existing job_dir, save_upload, _partition_dir ...

    def save_result(
        self,
        job: TranscriptionJob,
        transcript: TranscriptResult,
        file_metadata: FileMetadata,
    ) -> str:
        """Save transcript result with atomic writes.
        
        Uses temp files + rename pattern to ensure either:
        - Both files are written completely, or
        - Neither file exists (on failure)
        
        Raises:
            ValidationError: If job not completed
            StorageError: If write fails
        """
        if job.completed_at is None:
            raise ValidationError("Job must be completed before writing artifacts")

        artifact_dir = self._partition_dir(transcript.provider, job.completed_at, job.job_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        segments_path = artifact_dir / "segments.parquet"
        summary_path = artifact_dir / "summary.parquet"

        segment_rows = self._segment_rows(job, transcript, file_metadata)
        summary_row = self._summary_row(job, transcript, file_metadata, str(artifact_dir))

        # Write to temp files first
        try:
            self._atomic_write_parquet(segment_rows, segments_path)
            self._atomic_write_parquet([summary_row], summary_path)
        except Exception as exc:
            # Clean up any partial writes
            self._cleanup_partial_write(artifact_dir)
            raise StorageError(f"Failed to save transcript artifacts: {exc}") from exc

        return str(artifact_dir)

    def _atomic_write_parquet(self, rows: List[Dict[str, Any]], target_path: Path) -> None:
        """Write Parquet file atomically using temp file + rename.
        
        This ensures the target file is either:
        - Fully written and valid, or
        - Does not exist
        
        Never leaves a partial/corrupt file.
        """
        # Create temp file in same directory (same filesystem for atomic rename)
        fd, temp_path = tempfile.mkstemp(
            suffix=".parquet.tmp",
            dir=target_path.parent,
        )
        try:
            os.close(fd)  # Close the file descriptor, we'll use pyarrow
            
            # Write to temp file
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, temp_path)
            
            # Atomic rename (POSIX guarantees atomicity on same filesystem)
            os.replace(temp_path, target_path)
            
        except Exception:
            # Clean up temp file on any error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _cleanup_partial_write(self, artifact_dir: Path) -> None:
        """Remove any partial artifacts from failed write."""
        for filename in ["segments.parquet", "summary.parquet"]:
            try:
                (artifact_dir / filename).unlink(missing_ok=True)
            except OSError:
                pass
        # Remove temp files
        for temp_file in artifact_dir.glob("*.parquet.tmp"):
            try:
                temp_file.unlink()
            except OSError:
                pass

    # ... rest of methods unchanged ...
```

**Add StorageError to domain/errors.py:**
```python
class StorageError(AudioTranscriptError):
    """Raised when storage operations fail."""
```

**Test:**
```python
def test_atomic_write_cleanup_on_failure(tmp_path, mocker):
    """Verify partial writes are cleaned up on failure."""
    store = TranscriptArtifactStore(tmp_path / "art", tmp_path / "ds")
    
    # Make second write fail
    write_count = [0]
    original_write = pq.write_table
    def failing_write(*args, **kwargs):
        write_count[0] += 1
        if write_count[0] == 2:
            raise IOError("Disk full")
        return original_write(*args, **kwargs)
    
    mocker.patch("pyarrow.parquet.write_table", side_effect=failing_write)
    
    # ... setup job, transcript, metadata ...
    
    with pytest.raises(StorageError) as exc_info:
        store.save_result(job, transcript, metadata)
    
    # Verify no partial files left
    artifact_dir = tmp_path / "ds" / "provider=test" / ...
    assert not list(artifact_dir.glob("*.parquet"))
    assert not list(artifact_dir.glob("*.tmp"))
```

---

## 4.3 Move Import to Module Level

**File:** `src/audio_transcript/infra/repository.py`

**Problem:**
Line 195 has a delayed import inside a method:
```python
def _job_from_rows(self, row, attempt_rows):
    # ... code ...
    from ..domain.models import JobPayload
    job.payload = JobPayload(**job.payload)
```

Issues:
- Inconsistent with rest of codebase
- Small performance penalty on each call
- Hides dependencies
- Makes circular import issues harder to diagnose

**Solution:**
Move import to module level.

**Implementation:**

**Current (line 195):**
```python
def _job_from_rows(self, row: Dict[str, Any], attempt_rows: List[Dict[str, Any]]) -> TranscriptionJob:
    # ... code ...
    job = TranscriptionJob(
        job_id=row["job_id"],
        status=JobStatus(row["status"]),
        payload={
            "filename": row["source_filename"],
            # ...
        },
    )
    # dataclass payload construction using dict
    from ..domain.models import JobPayload  # <-- DELAYED IMPORT

    job.payload = JobPayload(**job.payload)
```

**Fixed (move to top of file):**
```python
"""Job repository implementations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import psycopg
from psycopg.rows import dict_row

from ..domain.errors import JobNotFoundError
from ..domain.models import (  # Consolidated imports
    FileMetadata,
    JobPayload,  # <-- MOVED HERE
    JobStatus,
    ProviderAttempt,
    TranscriptionJob,
)


# ... rest of file ...


def _job_from_rows(self, row: Dict[str, Any], attempt_rows: List[Dict[str, Any]]) -> TranscriptionJob:
    # ... code ...
    job = TranscriptionJob(
        job_id=row["job_id"],
        status=JobStatus(row["status"]),
        payload=JobPayload(  # <-- Direct construction, no dict intermediate
            filename=row["source_filename"],
            content_type=row["content_type"],
            source_path=row["source_path"],
            model_override=row["model_override"],
            chunk_duration_sec=row["chunk_duration_sec"],
            chunk_overlap_sec=row["chunk_overlap_sec"],
        ),
    )
    # No longer need the dict -> dataclass conversion
    # ... rest of method ...
```

**Benefits:**
- Clear dependencies at module level
- Slight performance improvement (no import lookup per call)
- Cleaner code - direct dataclass construction
- Consistent with codebase style

---

## Verification Checklist

After implementing Phase 4:

- [ ] `pytest tests/` passes
- [ ] Upload corrupt audio file → get `AudioProcessingError` with filename
- [ ] Simulate disk full during save → no partial Parquet files
- [ ] No delayed imports in repository.py
- [ ] Error responses include meaningful context for debugging
- [ ] 422 status code for audio processing failures

---

## Files Modified

| File | Change |
|------|--------|
| `src/audio_transcript/domain/errors.py` | Add AudioProcessingError, StorageError |
| `src/audio_transcript/services/audio.py` | Wrap subprocess errors |
| `src/audio_transcript/infra/storage.py` | Atomic writes with temp+rename |
| `src/audio_transcript/infra/repository.py` | Move JobPayload import to top |
| `src/audio_transcript/api/errors.py` | Add AudioProcessingError handler |

---

## Error Code Reference

After Phase 4, the API will return these error codes:

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `validation_error` | 400 | Invalid request parameters |
| `unauthorized` | 401 | Missing/invalid API key |
| `forbidden` | 403 | Authenticated but not authorized |
| `job_not_found` | 404 | Job ID doesn't exist |
| `result_unavailable` | 409 | Result not ready yet |
| `audio_processing_error` | 422 | ffmpeg/ffprobe failure |
| `application_error` | 500 | Known domain error |
| `internal_error` | 500 | Unexpected error |
