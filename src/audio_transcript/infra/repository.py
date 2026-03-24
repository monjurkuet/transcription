"""Job repository implementations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

try:
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover - exercised only when the pool extra is absent
    class ConnectionPool:
        """Small compatibility wrapper when psycopg_pool is unavailable."""

        def __init__(
            self,
            conninfo: str,
            min_size: int = 2,
            max_size: int = 10,
            timeout: int = 30,
            kwargs: Optional[Dict[str, Any]] = None,
        ):
            self.conninfo = conninfo
            self.min_size = min_size
            self.max_size = max_size
            self.timeout = timeout
            self.kwargs = kwargs or {}
            self._idle: List[Any] = []
            self._checked_out = 0

        @contextmanager
        def connection(self):
            conn = self._idle.pop() if self._idle else psycopg.connect(self.conninfo, **self.kwargs)
            self._checked_out += 1
            try:
                yield conn
            finally:
                self._checked_out -= 1
                if len(self._idle) < self.max_size:
                    self._idle.append(conn)
                else:
                    conn.close()

        def get_stats(self) -> Dict[str, int]:
            pool_size = len(self._idle) + self._checked_out
            return {
                "pool_size": max(pool_size, self.min_size if pool_size else 0),
                "pool_available": len(self._idle),
            }

        def close(self) -> None:
            while self._idle:
                self._idle.pop().close()

from ..domain.errors import JobNotFoundError
from ..domain.models import FileMetadata, JobPayload, JobStatus, ProviderAttempt, TranscriptionJob


class JobRepository(ABC):
    """Persistence interface for jobs."""

    @abstractmethod
    def create(self, job: TranscriptionJob) -> None:
        """Create a job."""

    @abstractmethod
    def get(self, job_id: str) -> TranscriptionJob:
        """Fetch a job."""

    @abstractmethod
    def save(self, job: TranscriptionJob) -> None:
        """Persist a job."""

    @abstractmethod
    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        provider: Optional[str] = None,
        filename: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
    ) -> List[TranscriptionJob]:
        """List jobs with optional filters."""

    @abstractmethod
    def healthcheck(self) -> Dict[str, str]:
        """Dependency readiness information."""


class PostgresJobRepository(JobRepository):
    """Postgres-backed durable job repository."""

    def __init__(self, database_url: str, min_size: int = 2, max_size: int = 10, timeout_sec: int = 30):
        self.database_url = database_url
        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout_sec,
            kwargs={"row_factory": dict_row},
        )
        self._ensure_schema()

    @contextmanager
    def _connection(self):
        with self._pool.connection() as conn:
            yield conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    model_override TEXT,
                    chunk_duration_sec INTEGER,
                    chunk_overlap_sec INTEGER,
                    provider TEXT,
                    model TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    error TEXT,
                    artifact_uri TEXT,
                    artifact_format TEXT,
                    summary_text TEXT,
                    segment_count INTEGER,
                    duration_sec DOUBLE PRECISION,
                    size_bytes BIGINT,
                    codec TEXT,
                    sample_rate INTEGER,
                    channels INTEGER,
                    bit_rate BIGINT,
                    format TEXT,
                    file_metadata JSONB
                );
                CREATE TABLE IF NOT EXISTS job_attempts (
                    id BIGSERIAL PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    provider TEXT NOT NULL,
                    key_id_masked TEXT,
                    success BOOLEAN NOT NULL,
                    retryable BOOLEAN NOT NULL,
                    error TEXT,
                    status_code INTEGER,
                    model TEXT,
                    latency_ms INTEGER,
                    started_at TIMESTAMPTZ NOT NULL,
                    finished_at TIMESTAMPTZ
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status_completed_at ON jobs(status, completed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_provider ON jobs(provider);
                CREATE INDEX IF NOT EXISTS idx_jobs_source_filename ON jobs(source_filename);
                CREATE INDEX IF NOT EXISTS idx_job_attempts_job_id ON job_attempts(job_id);
                """
            )
            self._ensure_attempt_constraints(cur)

    def create(self, job: TranscriptionJob) -> None:
        self.save(job)

    def get(self, job_id: str) -> TranscriptionJob:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            cur.execute(
                "SELECT provider, key_id_masked, success, retryable, error, status_code, model, latency_ms, started_at, finished_at FROM job_attempts WHERE job_id = %s ORDER BY id",
                (job_id,),
            )
            attempt_rows = cur.fetchall()
        return self._job_from_rows(row, attempt_rows)

    def save(self, job: TranscriptionJob) -> None:
        provider = self._final_provider(job)
        model = self._final_model(job)
        file_metadata = job.file_metadata.to_dict() if job.file_metadata else None
        summary_text = None
        segment_count = None
        if job.result_path:
            # summary fields are updated by the artifact store caller after save_result
            summary_text = getattr(job, "summary_text", None)
            segment_count = getattr(job, "segment_count", None)

        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (
                    job_id, status, source_filename, content_type, source_path, model_override,
                    chunk_duration_sec, chunk_overlap_sec, provider, model,
                    created_at, started_at, completed_at, error, artifact_uri, artifact_format,
                    summary_text, segment_count, duration_sec, size_bytes, codec, sample_rate,
                    channels, bit_rate, format, file_metadata
                ) VALUES (
                    %(job_id)s, %(status)s, %(source_filename)s, %(content_type)s, %(source_path)s, %(model_override)s,
                    %(chunk_duration_sec)s, %(chunk_overlap_sec)s, %(provider)s, %(model)s,
                    %(created_at)s, %(started_at)s, %(completed_at)s, %(error)s, %(artifact_uri)s, %(artifact_format)s,
                    %(summary_text)s, %(segment_count)s, %(duration_sec)s, %(size_bytes)s, %(codec)s, %(sample_rate)s,
                    %(channels)s, %(bit_rate)s, %(format)s, %(file_metadata)s
                )
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    source_filename = EXCLUDED.source_filename,
                    content_type = EXCLUDED.content_type,
                    source_path = EXCLUDED.source_path,
                    model_override = EXCLUDED.model_override,
                    chunk_duration_sec = EXCLUDED.chunk_duration_sec,
                    chunk_overlap_sec = EXCLUDED.chunk_overlap_sec,
                    provider = EXCLUDED.provider,
                    model = EXCLUDED.model,
                    created_at = EXCLUDED.created_at,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    error = EXCLUDED.error,
                    artifact_uri = EXCLUDED.artifact_uri,
                    artifact_format = EXCLUDED.artifact_format,
                    summary_text = COALESCE(EXCLUDED.summary_text, jobs.summary_text),
                    segment_count = COALESCE(EXCLUDED.segment_count, jobs.segment_count),
                    duration_sec = EXCLUDED.duration_sec,
                    size_bytes = EXCLUDED.size_bytes,
                    codec = EXCLUDED.codec,
                    sample_rate = EXCLUDED.sample_rate,
                    channels = EXCLUDED.channels,
                    bit_rate = EXCLUDED.bit_rate,
                    format = EXCLUDED.format,
                    file_metadata = EXCLUDED.file_metadata
                """,
                {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "source_filename": job.payload.filename,
                    "content_type": job.payload.content_type,
                    "source_path": job.payload.source_path,
                    "model_override": job.payload.model_override,
                    "chunk_duration_sec": job.payload.chunk_duration_sec,
                    "chunk_overlap_sec": job.payload.chunk_overlap_sec,
                    "provider": provider,
                    "model": model,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                    "error": job.error,
                    "artifact_uri": job.result_path,
                    "artifact_format": "parquet" if job.result_path else None,
                    "summary_text": summary_text,
                    "segment_count": segment_count,
                    "duration_sec": job.file_metadata.duration if job.file_metadata else None,
                    "size_bytes": job.file_metadata.size_bytes if job.file_metadata else None,
                    "codec": job.file_metadata.codec if job.file_metadata else None,
                    "sample_rate": job.file_metadata.sample_rate if job.file_metadata else None,
                    "channels": job.file_metadata.channels if job.file_metadata else None,
                    "bit_rate": job.file_metadata.bit_rate if job.file_metadata else None,
                    "format": job.file_metadata.format if job.file_metadata else None,
                    "file_metadata": json.dumps(file_metadata) if file_metadata else None,
                },
            )
            self._upsert_attempts(cur, job)

    def _ensure_attempt_constraints(self, cur) -> None:
        cur.execute(
            """
            DO $$
            BEGIN
                ALTER TABLE job_attempts
                ADD CONSTRAINT uq_job_attempts_job_provider_started
                UNIQUE (job_id, provider, started_at);
            EXCEPTION
                WHEN duplicate_object THEN NULL;
            END
            $$;
            """
        )

    def _upsert_attempts(self, cur, job: TranscriptionJob) -> None:
        for attempt in job.attempts:
            cur.execute(
                """
                INSERT INTO job_attempts (
                    job_id, provider, key_id_masked, success, retryable, error, status_code,
                    model, latency_ms, started_at, finished_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id, provider, started_at) DO UPDATE SET
                    key_id_masked = EXCLUDED.key_id_masked,
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

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        provider: Optional[str] = None,
        filename: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
    ) -> List[TranscriptionJob]:
        clauses = []
        params: List[Any] = []
        if status:
            clauses.append("status = %s")
            params.append(status)
        if provider:
            clauses.append("provider = %s")
            params.append(provider)
        if filename:
            clauses.append("source_filename ILIKE %s")
            params.append(f"%{filename}%")
        if search:
            clauses.append("summary_text ILIKE %s")
            params.append(f"%{search}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT %s",
                params,
            )
            rows = cur.fetchall()
            if not rows:
                return []
            job_ids = [row["job_id"] for row in rows]
            cur.execute(
                """
                SELECT job_id, provider, key_id_masked, success, retryable, error, status_code, model, latency_ms, started_at, finished_at
                FROM job_attempts
                WHERE job_id = ANY(%s)
                ORDER BY id
                """,
                (job_ids,),
            )
            attempt_rows = cur.fetchall()
        attempt_map: Dict[str, List[Dict[str, Any]]] = {job_id: [] for job_id in job_ids}
        for row in attempt_rows:
            attempt_map[row["job_id"]].append(row)
        return [self._job_from_rows(row, attempt_map.get(row["job_id"], [])) for row in rows]

    def healthcheck(self) -> Dict[str, str]:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        stats = self._pool.get_stats()
        return {
            "postgres": "ok",
            "postgres_pool_size": str(stats.get("pool_size", 0)),
            "postgres_pool_available": str(stats.get("pool_available", 0)),
        }

    def close(self) -> None:
        self._pool.close()

    def _job_from_rows(self, row: Dict[str, Any], attempt_rows: List[Dict[str, Any]]) -> TranscriptionJob:
        file_metadata = row["file_metadata"]
        if isinstance(file_metadata, str):
            file_metadata = json.loads(file_metadata)
        attempts = [
            ProviderAttempt(
                provider=item["provider"],
                key_id=item.get("key_id_masked"),
                started_at=item["started_at"],
                finished_at=item["finished_at"],
                success=item["success"],
                retryable=item["retryable"],
                error=item["error"],
                status_code=item["status_code"],
                model=item["model"],
                latency_ms=item["latency_ms"],
            )
            for item in attempt_rows
        ]
        job = TranscriptionJob(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            payload=JobPayload(
                filename=row["source_filename"],
                content_type=row["content_type"],
                source_path=row["source_path"],
                model_override=row["model_override"],
                chunk_duration_sec=row["chunk_duration_sec"],
                chunk_overlap_sec=row["chunk_overlap_sec"],
            ),
        )
        job.created_at = row["created_at"]
        job.started_at = row["started_at"]
        job.completed_at = row["completed_at"]
        job.error = row["error"]
        job.attempts = attempts
        job.result_path = row["artifact_uri"]
        job.file_metadata = FileMetadata(**file_metadata) if file_metadata else None
        job.summary_text = row["summary_text"]
        job.segment_count = row["segment_count"]
        job.provider = row["provider"]
        job.model = row["model"]
        return job

    def _final_provider(self, job: TranscriptionJob) -> Optional[str]:
        for attempt in reversed(job.attempts):
            if attempt.success:
                return attempt.provider
        return getattr(job, "provider", None)

    def _final_model(self, job: TranscriptionJob) -> Optional[str]:
        for attempt in reversed(job.attempts):
            if attempt.success and attempt.model:
                return attempt.model
        return getattr(job, "model", None)


class InMemoryJobRepository(JobRepository):
    """Test-friendly in-memory persistence."""

    def __init__(self):
        self.items: Dict[str, TranscriptionJob] = {}

    def create(self, job: TranscriptionJob) -> None:
        self.save(job)

    def get(self, job_id: str) -> TranscriptionJob:
        if job_id not in self.items:
            raise JobNotFoundError(job_id)
        return TranscriptionJob.from_record(self.items[job_id].to_record())

    def save(self, job: TranscriptionJob) -> None:
        self.items[job.job_id] = TranscriptionJob.from_record(job.to_record())

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        provider: Optional[str] = None,
        filename: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
    ) -> List[TranscriptionJob]:
        jobs = [self.get(job_id) for job_id in self.items]
        if status:
            jobs = [job for job in jobs if job.status.value == status]
        if provider:
            jobs = [job for job in jobs if getattr(job, "provider", None) == provider]
        if filename:
            jobs = [job for job in jobs if filename.lower() in job.payload.filename.lower()]
        if search:
            jobs = [job for job in jobs if search.lower() in (getattr(job, "summary_text", "") or "").lower()]
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return jobs[:limit]

    def healthcheck(self) -> Dict[str, str]:
        return {"memory_repository": "ok"}
