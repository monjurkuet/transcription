from audio_transcript.domain.models import TranscriptResult, TranscriptSegment
from audio_transcript.services.audio import merge_transcripts


def test_merge_transcripts_deduplicates_overlap():
    first = TranscriptResult(
        text="hello world",
        segments=[
            TranscriptSegment(start=0.0, end=2.0, text="hello"),
            TranscriptSegment(start=2.0, end=5.0, text="world"),
        ],
        provider="groq",
    )
    second = TranscriptResult(
        text="world again",
        segments=[
            TranscriptSegment(start=0.0, end=1.0, text="world"),
            TranscriptSegment(start=1.0, end=3.0, text="again"),
        ],
        provider="mistral",
    )

    merged = merge_transcripts([first, second], overlap_sec=1)

    assert [segment.text for segment in merged.segments] == ["hello", "world", "again"]
