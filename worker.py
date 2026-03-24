"""Worker process entrypoint."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_transcript.worker.runner import run_worker_loop


if __name__ == "__main__":
    run_worker_loop()
