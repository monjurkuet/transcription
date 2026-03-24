"""Logging helpers with request and job correlation support."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_job_id: ContextVar[Optional[str]] = ContextVar("job_id", default=None)

_RESERVED_LOG_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def set_request_context(request_id: str) -> Token:
    """Bind a request id to the current logging context."""
    return _request_id.set(request_id)


def clear_request_context(token: Token | None = None) -> None:
    """Clear or reset the current request id."""
    if token is None:
        _request_id.set(None)
        return
    _request_id.reset(token)


def set_job_context(job_id: str) -> Token:
    """Bind a job id to the current logging context."""
    return _job_id.set(job_id)


def clear_job_context(token: Token | None = None) -> None:
    """Clear or reset the current job id."""
    if token is None:
        _job_id.set(None)
        return
    _job_id.reset(token)


def clear_context() -> None:
    """Clear all request and job correlation context."""
    clear_request_context()
    clear_job_context()


def _context_fields() -> Dict[str, str]:
    fields: Dict[str, str] = {}
    request_id = _request_id.get()
    job_id = _job_id.get()
    if request_id:
        fields["request_id"] = request_id
    if job_id:
        fields["job_id"] = job_id
    return fields


def _extra_fields(record: logging.LogRecord) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED_LOG_FIELDS and not key.startswith("_")
    }


class StructuredFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_context_fields())
        payload.update(_extra_fields(record))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Render concise human-readable logs for local development."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        context = _context_fields()
        context_bits = []
        if "request_id" in context:
            context_bits.append(f"req={context['request_id'][:8]}")
        if "job_id" in context:
            context_bits.append(f"job={context['job_id'][:8]}")
        context_str = ""
        if context_bits:
            context_str = f" [{', '.join(context_bits)}]"

        extras = _extra_fields(record)
        extra_keys: Iterable[str] = sorted(extras)
        extra_str = ""
        if extras:
            extra_str = " (" + ", ".join(f"{key}={extras[key]}" for key in extra_keys) + ")"

        message = f"{timestamp} {record.levelname[0]} {record.name}: {record.getMessage()}{context_str}{extra_str}"
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        return message


def configure_logging(
    level: str = "INFO",
    log_format: str = "text",
    *,
    json_format: bool | None = None,
    stream=None,
) -> None:
    """Configure root logging for API and worker processes."""
    if json_format is not None:
        log_format = "json" if json_format else "text"

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(resolved_level)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(resolved_level)
    if log_format == "json":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(HumanFormatter())
    root.addHandler(handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
