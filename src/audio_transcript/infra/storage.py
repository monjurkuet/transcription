"""Artifact storage for uploads and transcript Parquet datasets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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

    def job_dir(self, job_id: str) -> Path:
        """Return the upload directory for a job, creating it if needed."""
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_upload(self, job_id: str, file_storage: FileStorage) -> Path:
        """Persist an uploaded source file under the job artifact directory."""
        filename = secure_filename(file_storage.filename or "")
        if not filename:
            raise ValidationError("A filename is required")
        path = self.job_dir(job_id) / filename
        file_storage.save(path)
        return path

    def _partition_dir(self, provider: str, completed_at, job_id: str) -> Path:
        return (
            self.dataset_root
            / f"provider={provider}"
            / f"year={completed_at:%Y}"
            / f"month={completed_at:%m}"
            / f"day={completed_at:%d}"
            / job_id
        )

    def save_result(
        self,
        job: TranscriptionJob,
        transcript: TranscriptResult,
        file_metadata: FileMetadata,
    ) -> str:
        """Persist transcript summary and segment artifacts as Parquet files."""
        if job.completed_at is None:
            raise ValidationError("Job must be completed before writing artifacts")

        artifact_dir = self._partition_dir(transcript.provider, job.completed_at, job.job_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        segments_path = artifact_dir / "segments.parquet"
        summary_path = artifact_dir / "summary.parquet"

        segment_rows = self._segment_rows(job, transcript, file_metadata)
        summary_row = self._summary_row(job, transcript, file_metadata, str(artifact_dir))
        try:
            self._atomic_write_parquet(segment_rows, segments_path)
            self._atomic_write_parquet([summary_row], summary_path)
        except Exception as exc:
            self._cleanup_partial_write(artifact_dir)
            raise StorageError("Failed to persist transcript artifacts") from exc
        return str(artifact_dir)

    def _atomic_write_parquet(self, rows: List[Dict[str, Any]], target_path: Path) -> None:
        fd, temp_path = tempfile.mkstemp(suffix=".parquet.tmp", dir=target_path.parent)
        os.close(fd)
        try:
            pq.write_table(pa.Table.from_pylist(rows), temp_path)
            os.replace(temp_path, target_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _cleanup_partial_write(self, artifact_dir: Path) -> None:
        for filename in ("segments.parquet", "summary.parquet"):
            try:
                (artifact_dir / filename).unlink(missing_ok=True)
            except OSError:
                pass
        for temp_file in artifact_dir.glob("*.parquet.tmp"):
            try:
                temp_file.unlink()
            except OSError:
                pass

    def load_result(
        self,
        artifact_uri: str | Path,
        segment_offset: int = 0,
        segment_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load a stored result document with optional segment pagination."""
        artifact_dir = Path(artifact_uri)
        if artifact_dir.is_file() and artifact_dir.suffix.lower() == ".json":
            with open(artifact_dir, "r", encoding="utf-8") as handle:
                return json.load(handle)
        segments_path = artifact_dir / "segments.parquet"
        summary_path = artifact_dir / "summary.parquet"
        if not segments_path.exists() or not summary_path.exists():
            raise ArtifactNotFoundError(f"Missing Parquet artifact under {artifact_dir}")

        summary_rows = pq.ParquetFile(summary_path).read().to_pylist()
        if not summary_rows:
            raise ArtifactNotFoundError(f"Empty summary artifact under {artifact_dir}")
        summary = summary_rows[0]
        segments = list(self._iter_segments(segments_path, offset=segment_offset, limit=segment_limit))
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
            },
            "attempts": json.loads(summary["attempts_json"]),
        }
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
    ) -> Iterator[Dict[str, Any]]:
        """Iterate stored segment rows while applying pagination offsets."""
        yielded = 0
        current_index = 0
        max_items = limit if limit is not None else None
        parquet = pq.ParquetFile(segments_path)

        for batch in parquet.iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                if row.get("segment_index", -1) < 0:
                    continue
                if current_index < offset:
                    current_index += 1
                    continue
                if max_items is not None and yielded >= max_items:
                    return
                segment = self._segment_to_result_item(current_index, row)
                if segment:
                    yield segment
                    yielded += 1
                current_index += 1

    def _segment_rows(
        self,
        job: TranscriptionJob,
        transcript: TranscriptResult,
        file_metadata: FileMetadata,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index, segment in enumerate(transcript.segments):
            rows.append(
                {
                    "job_id": job.job_id,
                    "segment_index": index,
                    "segment_id": segment.id,
                    "start_sec": segment.start,
                    "end_sec": segment.end,
                    "text": segment.text,
                    "provider": transcript.provider,
                    "model": transcript.model,
                    "created_at": job.created_at.isoformat(),
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "source_filename": job.payload.filename,
                    "duration_sec": file_metadata.duration,
                    "avg_logprob": segment.provider_data.get("avg_logprob"),
                    "compression_ratio": segment.provider_data.get("compression_ratio"),
                    "no_speech_prob": segment.provider_data.get("no_speech_prob"),
                    "provider_data_json": json.dumps(segment.provider_data, ensure_ascii=False),
                }
            )
        if rows:
            return rows
        return [
            {
                "job_id": job.job_id,
                "segment_index": -1,
                "segment_id": None,
                "start_sec": None,
                "end_sec": None,
                "text": None,
                "provider": transcript.provider,
                "model": transcript.model,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "source_filename": job.payload.filename,
                "duration_sec": file_metadata.duration,
                "avg_logprob": None,
                "compression_ratio": None,
                "no_speech_prob": None,
                "provider_data_json": json.dumps({}, ensure_ascii=False),
            }
        ]

    def _summary_row(
        self,
        job: TranscriptionJob,
        transcript: TranscriptResult,
        file_metadata: FileMetadata,
        artifact_uri: str,
    ) -> Dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "provider": transcript.provider,
            "model": transcript.model,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "source_filename": job.payload.filename,
            "text": transcript.text,
            "summary_text": transcript.text[:1000],
            "segment_count": len(transcript.segments),
            "artifact_uri": artifact_uri,
            "attempts_json": json.dumps([attempt.to_dict() for attempt in job.attempts], ensure_ascii=False),
            "file_metadata_filename": file_metadata.filename,
            "file_metadata_path": file_metadata.path,
            "duration_sec": file_metadata.duration,
            "size_bytes": file_metadata.size_bytes,
            "format": file_metadata.format,
            "bit_rate": file_metadata.bit_rate,
            "codec": file_metadata.codec,
            "sample_rate": file_metadata.sample_rate,
            "channels": file_metadata.channels,
        }

    def _segment_to_result_item(self, index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        provider_data = json.loads(row["provider_data_json"]) if row.get("provider_data_json") else {}
        if row.get("segment_index", -1) < 0:
            return {}
        data: Dict[str, Any] = {
            "start": row["start_sec"],
            "end": row["end_sec"],
            "text": row["text"],
        }
        if row.get("segment_id") is not None:
            data["id"] = row["segment_id"]
        if provider_data:
            data["provider_data"] = provider_data
        return data
