"""HTTP routes."""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from ..domain.errors import ValidationError
from ..domain.models import JobPayload, JobStatus, TranscriptionJob, is_supported_audio_file
from ..infra.storage import TranscriptArtifactStore
from .auth import require_api_key


bp = Blueprint("v1", __name__, url_prefix="/v1")


def _check_binary(name: str) -> str:
    """Return a machine-readable health status for a local binary."""
    if shutil.which(name) is None:
        return "not_found"
    try:
        proc = subprocess.run(
            [name, "-version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    except OSError:
        return "error"
    return "ok" if proc.returncode == 0 else "error"


@bp.get("/health")
def health():
    """Return application readiness and optional deep media-tool checks."""
    repo = current_app.config["repository"]
    queue = current_app.config["queue"]
    runtime_state = current_app.config["runtime_state"]
    status = {"status": "ok"}
    status.update(repo.healthcheck())
    status.update(queue.healthcheck())
    status.update(runtime_state.healthcheck())
    if request.args.get("deep", "").lower() == "true":
        status["ffmpeg"] = _check_binary("ffmpeg")
        status["ffprobe"] = _check_binary("ffprobe")
    if any(value != "ok" for key, value in status.items() if key != "status"):
        status["status"] = "degraded"
    return jsonify(status)


@bp.get("/providers/status")
@require_api_key
def provider_status():
    providers = current_app.config["providers"]
    payload = {"providers": [provider.status() for provider in providers.values()]}
    fallback_provider = current_app.config.get("fallback_provider")
    if fallback_provider is not None:
        payload["providers"].append(fallback_provider.status())
    return jsonify(payload)


@bp.post("/jobs")
@require_api_key
def create_job():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": {"code": "validation_error", "message": "file is required", "details": {}}}), 400
    if not is_supported_audio_file(Path(uploaded.filename or "")):
        return (
            jsonify({"error": {"code": "validation_error", "message": "unsupported audio file", "details": {}}}),
            400,
        )

    job_id = str(uuid.uuid4())
    store: TranscriptArtifactStore = current_app.config["artifact_store"]
    saved_path = store.save_upload(job_id, uploaded)
    payload = JobPayload(
        filename=uploaded.filename or saved_path.name,
        content_type=uploaded.mimetype or "application/octet-stream",
        source_path=str(saved_path),
        model_override=request.form.get("model") or None,
        chunk_duration_sec=int(request.form["chunk_duration_sec"]) if request.form.get("chunk_duration_sec") else None,
        chunk_overlap_sec=int(request.form["chunk_overlap_sec"]) if request.form.get("chunk_overlap_sec") else None,
    )
    job = TranscriptionJob(job_id=job_id, status=JobStatus.QUEUED, payload=payload)
    current_app.config["service"].create_job(job)
    current_app.config["queue"].enqueue(job_id)
    return (
        jsonify(
            {
                "job": {
                    "id": job_id,
                    "status": job.status.value,
                    "filename": payload.filename,
                },
                "links": {
                    "status": f"/v1/jobs/{job_id}",
                    "result": f"/v1/jobs/{job_id}/result",
                },
            }
        ),
        202,
    )


@bp.get("/jobs/<job_id>")
@require_api_key
def get_job(job_id: str):
    job = current_app.config["repository"].get(job_id)
    return jsonify(job.public_dict())


@bp.get("/jobs")
@require_api_key
def list_jobs():
    jobs = current_app.config["repository"].list_jobs(
        status=request.args.get("status"),
        provider=request.args.get("provider"),
        filename=request.args.get("filename"),
        search=request.args.get("search"),
        limit=int(request.args.get("limit", 50)),
    )
    return jsonify({"jobs": [job.public_dict()["job"] | {"provider": job.provider, "segment_count": job.segment_count, "summary_text": job.summary_text} for job in jobs]})


@bp.get("/jobs/<job_id>/result")
@require_api_key
def get_result(job_id: str):
    job = current_app.config["repository"].get(job_id)
    if job.status != JobStatus.SUCCEEDED or not job.result_path:
        return (
            jsonify({"error": {"code": "result_unavailable", "message": "Result is not ready", "details": {}}}),
            409,
        )
    try:
        segment_offset = int(request.args.get("segment_offset", 0))
    except ValueError as exc:
        raise ValidationError("segment_offset must be an integer") from exc
    limit_arg = request.args.get("segment_limit")
    try:
        segment_limit = int(limit_arg) if limit_arg is not None else None
    except ValueError as exc:
        raise ValidationError("segment_limit must be an integer") from exc
    if segment_offset < 0:
        raise ValidationError("segment_offset must be >= 0")
    if segment_limit is not None and segment_limit <= 0:
        raise ValidationError("segment_limit must be > 0")
    document = current_app.config["artifact_store"].load_result(
        job.result_path,
        segment_offset=segment_offset,
        segment_limit=segment_limit,
    )
    return jsonify(document)
