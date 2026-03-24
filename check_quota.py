#!/usr/bin/env python3
"""Provider status CLI."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_transcript.api.app import build_runtime
from audio_transcript.config import Settings


def main() -> None:
    settings = Settings.from_env()
    runtime = build_runtime(settings)
    payload = {"providers": [provider.status() for provider in runtime["providers"].values()]}
    fallback = runtime.get("fallback_provider")
    if fallback is not None:
        payload["providers"].append(fallback.status())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
