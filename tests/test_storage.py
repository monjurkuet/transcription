from audio_transcript.domain.models import FileMetadata, JobPayload, JobStatus, TranscriptResult, TranscriptSegment, TranscriptionJob
from audio_transcript.infra.storage import TranscriptArtifactStore


def make_file_metadata(path):
    return FileMetadata(
        filename=path.name,
        path=str(path),
        size_bytes=path.stat().st_size,
        duration=1.0,
        format=path.suffix.lstrip("."),
        bit_rate=0,
        codec="pcm",
        sample_rate=16000,
        channels=1,
    )


def test_load_result_with_pagination(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    store = TranscriptArtifactStore(tmp_path / "artifacts", tmp_path / "dataset")
    job = TranscriptionJob(
        job_id="job-1",
        status=JobStatus.SUCCEEDED,
        payload=JobPayload(filename=audio_path.name, content_type="audio/wav", source_path=str(audio_path)),
    )
    job.completed_at = job.created_at
    transcript = TranscriptResult(
        text="joined",
        segments=[
            TranscriptSegment(start=float(i), end=float(i + 1), text=f"segment-{i}")
            for i in range(5)
        ],
        provider="groq",
    )
    artifact_uri = store.save_result(job, transcript, make_file_metadata(audio_path))

    first = store.load_result(artifact_uri, segment_offset=0, segment_limit=2)
    second = store.load_result(artifact_uri, segment_offset=2, segment_limit=2)

    assert [item["text"] for item in first["transcript"]["segments"]] == ["segment-0", "segment-1"]
    assert first["pagination"] == {"offset": 0, "limit": 2, "total": 5, "has_more": True}
    assert [item["text"] for item in second["transcript"]["segments"]] == ["segment-2", "segment-3"]
    assert second["pagination"] == {"offset": 2, "limit": 2, "total": 5, "has_more": True}


def test_load_result_skips_placeholder_segment_rows(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFfake")
    store = TranscriptArtifactStore(tmp_path / "artifacts", tmp_path / "dataset")
    job = TranscriptionJob(
        job_id="job-empty",
        status=JobStatus.SUCCEEDED,
        payload=JobPayload(filename=audio_path.name, content_type="audio/wav", source_path=str(audio_path)),
    )
    job.completed_at = job.created_at
    transcript = TranscriptResult(text="", segments=[], provider="groq")
    artifact_uri = store.save_result(job, transcript, make_file_metadata(audio_path))

    result = store.load_result(artifact_uri)

    assert result["transcript"]["segments"] == []
