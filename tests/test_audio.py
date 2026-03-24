import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from audio_transcript.domain.errors import AudioProcessingError
from audio_transcript.services.audio import AudioChunker, AudioInspector


def test_audio_inspector_handles_invalid_file(tmp_path):
    bad_file = tmp_path / "bad.wav"
    bad_file.write_bytes(b"not audio")
    inspector = AudioInspector()

    with patch(
        "audio_transcript.services.audio.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["ffprobe"], stderr="Invalid data found when processing input"),
    ):
        with pytest.raises(AudioProcessingError) as exc_info:
            inspector.get_file_metadata(bad_file)

    assert "bad.wav" in str(exc_info.value)
    assert "Failed to read audio metadata" in str(exc_info.value)


def test_audio_inspector_handles_invalid_duration_output(tmp_path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake")
    inspector = AudioInspector()

    class Result:
        stdout = "not-a-number"

    with patch("audio_transcript.services.audio.subprocess.run", return_value=Result()):
        with pytest.raises(AudioProcessingError) as exc_info:
            inspector.get_duration(sample)

    assert "Invalid duration value" in str(exc_info.value)


def test_audio_chunker_wraps_ffmpeg_failure(tmp_path):
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake")
    chunker = AudioChunker(AudioInspector())

    with patch(
        "audio_transcript.services.audio.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["ffmpeg"], stderr="Conversion failed"),
    ):
        with pytest.raises(AudioProcessingError) as exc_info:
            chunker._create_chunk(sample, tmp_path / "out.wav", 0, 1)

    assert "Failed to create chunk" in str(exc_info.value)
    assert "Conversion failed" in str(exc_info.value)
