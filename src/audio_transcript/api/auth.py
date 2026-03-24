"""Authentication helpers."""

from __future__ import annotations

from functools import wraps
from typing import Callable

from flask import current_app, request

from ..domain.errors import ValidationError


def require_api_key(view: Callable):
    """Require the configured inbound API key."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        expected = current_app.config["settings"].service_api_key
        if request.headers.get("X-API-Key") != expected:
            raise ValidationError("Unauthorized")
        return view(*args, **kwargs)

    return wrapper
