"""Flask error handlers."""

from __future__ import annotations

import logging

from flask import jsonify

from ..domain.errors import AudioProcessingError, AudioTranscriptError, JobNotFoundError, StorageError, ValidationError

logger = logging.getLogger("audio_transcript.api.errors")


def register_error_handlers(app) -> None:
    """Register consistent JSON errors."""

    @app.errorhandler(JobNotFoundError)
    def handle_not_found(exc):
        return jsonify({"error": {"code": "job_not_found", "message": str(exc), "details": {}}}), 404

    @app.errorhandler(ValidationError)
    def handle_validation(exc):
        status = 401 if str(exc) == "Unauthorized" else 400
        code = "unauthorized" if status == 401 else "validation_error"
        return jsonify({"error": {"code": code, "message": str(exc), "details": {}}}), status

    @app.errorhandler(AudioProcessingError)
    def handle_audio_processing(exc):
        return jsonify({"error": {"code": "audio_processing_error", "message": str(exc), "details": {}}}), 422

    @app.errorhandler(StorageError)
    def handle_storage(exc):
        logger.exception("Storage error")
        return jsonify(
            {"error": {"code": "storage_error", "message": "Failed to persist transcript artifacts", "details": {}}}
        ), 500

    @app.errorhandler(AudioTranscriptError)
    def handle_domain(exc):
        logger.exception("Application error")
        return jsonify({"error": {"code": "application_error", "message": "An application error occurred", "details": {}}}), 500

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        logger.exception("Unexpected error")
        return jsonify({"error": {"code": "internal_error", "message": "An unexpected error occurred", "details": {}}}), 500
