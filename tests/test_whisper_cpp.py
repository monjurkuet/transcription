from unittest.mock import Mock, patch

import requests

from audio_transcript.domain.errors import RetryableProviderError

from audio_transcript.infra.providers.whisper_cpp import WhisperCppProvider


def test_whisper_cpp_load_model_posts_model_path():
    provider = WhisperCppProvider("http://127.0.0.1:8334", 30, 0.0, 0.2)
    response = Mock(status_code=200, text="ok")

    with patch("audio_transcript.infra.providers.whisper_cpp.requests.post", return_value=response) as mock_post:
        provider.load_model("/models/ggml-base.bin")

    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["files"] == {"model": (None, "/models/ggml-base.bin")}


def test_whisper_cpp_request_error_is_retryable(tmp_path):
    provider = WhisperCppProvider("http://127.0.0.1:8334", 30, 0.0, 0.2)
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake")

    with patch(
        "audio_transcript.infra.providers.whisper_cpp.requests.post",
        side_effect=requests.exceptions.ConnectionError("boom"),
    ):
        try:
            provider.transcribe(sample, "audio/wav")
        except RetryableProviderError as exc:
            assert "boom" in str(exc)
        else:
            raise AssertionError("expected RetryableProviderError")


def test_whisper_cpp_uses_shared_segment_parser(tmp_path):
    provider = WhisperCppProvider("http://127.0.0.1:8334", 30, 0.0, 0.2)
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake")
    response = Mock(status_code=200, text="ok", headers={})
    response.json.return_value = {
        "result": "hello",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "avg_logprob": -0.5,
            }
        ],
    }

    with patch("audio_transcript.infra.providers.whisper_cpp.requests.post", return_value=response):
        result = provider.transcribe(sample, "audio/wav")

    assert result.text == "hello"
    assert result.segments[0].provider_data == {"avg_logprob": -0.5}
