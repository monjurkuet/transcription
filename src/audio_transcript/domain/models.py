"""Domain models for the transcription service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def isoformat_or_none(value: Optional[datetime]) -> Optional[str]:
    """Serialize an optional datetime."""
    return value.isoformat() if value else None


class JobStatus(str, Enum):
    """Job lifecycle states."""

    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class FileMetadata:
    """Normalized source file metadata extracted from ffprobe output."""

    filename: str
    path: str
    size_bytes: int
    duration: float
    format: str
    bit_rate: int
    codec: str
    sample_rate: int
    channels: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata for API and artifact persistence."""
        return asdict(self)


@dataclass
class TranscriptSegment:
    """A normalized transcript segment."""

    start: float
    end: float
    text: str
    id: Optional[int] = None
    provider_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.id is not None:
            data["id"] = self.id
        if self.provider_data:
            data["provider_data"] = self.provider_data
        return data


@dataclass
class TranscriptResult:
    """Normalized transcript output returned by a provider adapter."""

    text: str
    segments: List[TranscriptSegment]
    provider: str
    model: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transcript content for API and artifact payloads."""
        data = {
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
            "provider": self.provider,
        }
        if self.model:
            data["model"] = self.model
        return data


@dataclass
class ProviderAttempt:
    """Audit trail for a single provider attempt against one audio input."""

    provider: str
    started_at: datetime
    key_id: Optional[str] = None
    finished_at: Optional[datetime] = None
    success: bool = False
    retryable: bool = False
    error: Optional[str] = None
    status_code: Optional[int] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize provider attempt metadata for storage and APIs."""
        return {
            "provider": self.provider,
            "key_id": self.key_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": isoformat_or_none(self.finished_at),
            "success": self.success,
            "retryable": self.retryable,
            "error": self.error,
            "status_code": self.status_code,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


@dataclass
class JobPayload:
    """Job configuration captured from the original API request."""

    filename: str
    content_type: str
    source_path: str
    model_override: Optional[str] = None
    chunk_duration_sec: Optional[int] = None
    chunk_overlap_sec: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the original request payload for persistence."""
        return asdict(self)


@dataclass
class TranscriptionJob:
    """Persisted job state shared between API, worker, and storage layers."""

    job_id: str
    status: JobStatus
    payload: JobPayload
    created_at: datetime = field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    attempts: List[ProviderAttempt] = field(default_factory=list)
    result_path: Optional[str] = None
    file_metadata: Optional[FileMetadata] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    summary_text: Optional[str] = None
    segment_count: Optional[int] = None

    def to_record(self) -> Dict[str, Any]:
        """Serialize the full job record for repository implementations."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "payload": self.payload.to_dict(),
            "created_at": self.created_at.isoformat(),
            "started_at": isoformat_or_none(self.started_at),
            "completed_at": isoformat_or_none(self.completed_at),
            "error": self.error,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "result_path": self.result_path,
            "file_metadata": self.file_metadata.to_dict() if self.file_metadata else None,
            "provider": self.provider,
            "model": self.model,
            "summary_text": self.summary_text,
            "segment_count": self.segment_count,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "TranscriptionJob":
        """Hydrate a persisted record back into a strongly typed job model."""
        payload = JobPayload(**record["payload"])
        attempts = [
            ProviderAttempt(
                provider=item["provider"],
                key_id=item.get("key_id"),
                started_at=datetime.fromisoformat(item["started_at"]),
                finished_at=datetime.fromisoformat(item["finished_at"]) if item.get("finished_at") else None,
                success=item.get("success", False),
                retryable=item.get("retryable", False),
                error=item.get("error"),
                status_code=item.get("status_code"),
                model=item.get("model"),
                latency_ms=item.get("latency_ms"),
            )
            for item in record.get("attempts", [])
        ]
        file_metadata = record.get("file_metadata")
        return cls(
            job_id=record["job_id"],
            status=JobStatus(record["status"]),
            payload=payload,
            created_at=datetime.fromisoformat(record["created_at"]),
            started_at=datetime.fromisoformat(record["started_at"]) if record.get("started_at") else None,
            completed_at=datetime.fromisoformat(record["completed_at"]) if record.get("completed_at") else None,
            error=record.get("error"),
            attempts=attempts,
            result_path=record.get("result_path"),
            file_metadata=FileMetadata(**file_metadata) if file_metadata else None,
            provider=record.get("provider"),
            model=record.get("model"),
            summary_text=record.get("summary_text"),
            segment_count=record.get("segment_count"),
        )

    def public_dict(self) -> Dict[str, Any]:
        """Build the public API representation returned by job endpoints."""
        return {
            "job": {
                "id": self.job_id,
                "status": self.status.value,
                "filename": self.payload.filename,
                "provider": self.provider,
                "model": self.model,
                "segment_count": self.segment_count,
                "summary_text": self.summary_text,
                "created_at": self.created_at.isoformat(),
                "started_at": isoformat_or_none(self.started_at),
                "completed_at": isoformat_or_none(self.completed_at),
                "error": self.error,
            },
            "file_metadata": self.file_metadata.to_dict() if self.file_metadata else None,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass
class ProviderQuotaState:
    """Provider key availability exposed by the status endpoint."""

    provider: str
    key_id: str
    available: bool
    cooldown_until: Optional[datetime] = None
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize provider quota state for the status endpoint."""
        return {
            "provider": self.provider,
            "key_id": self.key_id,
            "available": self.available,
            "cooldown_until": isoformat_or_none(self.cooldown_until),
            "last_error": self.last_error,
        }


def build_result_document(
    job: TranscriptionJob,
    transcript: TranscriptResult,
    file_metadata: FileMetadata,
) -> Dict[str, Any]:
    """Create the result document shape served by the result endpoint."""
    return {
        "job": {
            "id": job.job_id,
            "status": job.status.value,
            "filename": job.payload.filename,
            "created_at": job.created_at.isoformat(),
            "started_at": isoformat_or_none(job.started_at),
            "completed_at": isoformat_or_none(job.completed_at),
        },
        "file_metadata": file_metadata.to_dict(),
        "transcript": transcript.to_dict(),
        "attempts": [attempt.to_dict() for attempt in job.attempts],
    }


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".flac",
    ".m4a",
    ".ogg",
    ".webm",
    ".mpeg",
    ".mpga",
}


def is_supported_audio_file(path: Path) -> bool:
    """Return whether the path is a supported audio file."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS
