#!/usr/bin/env python3
"""Development server entrypoint."""

from app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
