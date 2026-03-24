from unittest.mock import Mock, patch

import requests

from audio_transcript.domain.errors import NonRetryableProviderError, RetryableProviderError
from audio_transcript.infra.providers.base import parse_segments
from audio_transcript.infra.providers.groq import GroqProvider
from audio_transcript.infra.providers.mistral import MistralProvider
from audio_transcript.infra.runtime_state import InMemoryRuntimeState
from audio_transcript.services.router import ProviderKeyPool


def make_response(status_code=200, text="ok", payload=None, headers=None):
    response = Mock(status_code=status_code, text=text, headers=headers or {})
    response.json.return_value = payload or {}
    return response


def test_parse_segments_extracts_standard_fields_and_exclusions():
    segments = parse_segments(
        {
            "segments": [
                {
                    "id": 1,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "tokens": [1, 2, 3],
                    "avg_logprob": -0.5,
                }
            ]
        },
        excluded_keys={"tokens"},
    )

    assert len(segments) == 1
    assert segments[0].id == 1
    assert segments[0].text == "hello"
    assert segments[0].provider_data == {"avg_logprob": -0.5}


def test_groq_provider_excludes_tokens_from_provider_data(tmp_path):
    provider = GroqProvider(ProviderKeyPool("groq", ["gsk_test_key"]), "whisper-large-v3", 30)
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake")
    response = make_response(
        payload={
            "text": "hello",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "tokens": [1, 2],
                    "avg_logprob": -0.25,
                }
            ],
        }
    )

    with patch("audio_transcript.infra.providers.base.requests.post", return_value=response):
        result = provider.transcribe(sample, "audio/wav", model_override="override-model")

    assert result.model == "override-model"
    assert result.text == "hello"
    assert result.segments[0].provider_data == {"avg_logprob": -0.25}


def test_mistral_provider_keeps_non_standard_segment_fields(tmp_path):
    provider = MistralProvider(ProviderKeyPool("mistral", ["ms_test_key"]), "voxtral-mini-latest", 30)
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake")
    response = make_response(
        payload={
            "text": "hello",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "avg_logprob": -0.25,
                    "temperature": 0.0,
                }
            ],
        }
    )

    with patch("audio_transcript.infra.providers.base.requests.post", return_value=response):
        result = provider.transcribe(sample, "audio/wav")

    assert result.model == "voxtral-mini-latest"
    assert result.segments[0].provider_data == {"avg_logprob": -0.25, "temperature": 0.0}


def test_remote_provider_429_triggers_cooldown(tmp_path):
    runtime_state = InMemoryRuntimeState()
    pool = ProviderKeyPool("groq", ["gsk_test_key"], runtime_state=runtime_state)
    provider = GroqProvider(pool, "whisper-large-v3", 30)
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake")
    response = make_response(status_code=429, text="rate limited", headers={"retry-after": "7"})

    with patch("audio_transcript.infra.providers.base.requests.post", return_value=response):
        try:
            provider.transcribe(sample, "audio/wav")
        except RetryableProviderError as exc:
            assert "429" in str(exc)
        else:
            raise AssertionError("expected RetryableProviderError")

    statuses = provider.status()["keys"]
    assert len(statuses) == 1
    assert statuses[0]["available"] is False
    assert statuses[0]["last_error"] is not None


def test_remote_provider_request_failure_is_retryable(tmp_path):
    provider = MistralProvider(ProviderKeyPool("mistral", ["ms_test_key"]), "voxtral-mini-latest", 30)
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake")

    with patch(
        "audio_transcript.infra.providers.base.requests.post",
        side_effect=requests.exceptions.ConnectionError("boom"),
    ):
        try:
            provider.transcribe(sample, "audio/wav")
        except RetryableProviderError as exc:
            assert "boom" in str(exc)
        else:
            raise AssertionError("expected RetryableProviderError")


def test_remote_provider_client_error_is_non_retryable(tmp_path):
    provider = MistralProvider(ProviderKeyPool("mistral", ["ms_test_key"]), "voxtral-mini-latest", 30)
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake")
    response = make_response(status_code=422, text="bad request")

    with patch("audio_transcript.infra.providers.base.requests.post", return_value=response):
        try:
            provider.transcribe(sample, "audio/wav")
        except NonRetryableProviderError as exc:
            assert "422" in str(exc)
        else:
            raise AssertionError("expected NonRetryableProviderError")
