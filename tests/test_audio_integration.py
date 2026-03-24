import wave

import pytest

from audio_transcript.services.audio import AudioChunker, AudioInspector
from conftest import ffmpeg_available

pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not available"),
]


@pytest.fixture
def sample_wav(tmp_path):
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000 * 5)
    return wav_path


@pytest.fixture
def long_wav(tmp_path):
    wav_path = tmp_path / "long.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000 * 35)
    return wav_path


def test_audio_inspector_integration(sample_wav):
    inspector = AudioInspector()
    duration = inspector.get_duration(sample_wav)
    metadata = inspector.get_file_metadata(sample_wav)
    assert abs(duration - 5.0) < 0.2
    assert metadata.channels == 1
    assert metadata.sample_rate == 16000
    assert metadata.size_bytes > 0


def test_audio_chunker_integration(long_wav, tmp_path):
    inspector = AudioInspector()
    chunker = AudioChunker(inspector)
    chunk_dir = tmp_path / "chunks"
    chunks = chunker.chunk_audio(long_wav, chunk_dir, duration_sec=10, overlap_sec=2)
    assert len(chunks) >= 4
    for chunk in chunks:
        assert chunk.exists()
        metadata = inspector.get_file_metadata(chunk)
        assert metadata.channels == 1
        assert metadata.sample_rate == 16000
        assert metadata.duration > 0
