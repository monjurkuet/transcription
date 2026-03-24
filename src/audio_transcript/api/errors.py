"""Flask error handlers."""

from __future__ import annotations

import logging

from flask import jsonify

from ..domain.errors import AudioTranscriptError, JobNotFoundError, ValidationError

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

    @app.errorhandler(AudioTranscriptError)
    def handle_domain(exc):
        logger.exception("Application error")
        return jsonify({"error": {"code": "application_error", "message": "An application error occurred", "details": {}}}), 500

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        logger.exception("Unexpected error")
        return jsonify({"error": {"code": "internal_error", "message": "An unexpected error occurred", "details": {}}}), 500
