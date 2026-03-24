"""Audio inspection and chunking helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List

from ..domain.errors import AudioProcessingError, ValidationError
from ..domain.models import FileMetadata, TranscriptResult, TranscriptSegment


class AudioInspector:
    """Audio metadata utilities backed by ffprobe."""

    def _subprocess_message(self, exc: subprocess.CalledProcessError) -> str:
        output = exc.stderr or exc.stdout or "Unknown error"
        return output.strip() or "Unknown error"

    def get_duration(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except subprocess.CalledProcessError as exc:
            raise AudioProcessingError(
                f"Failed to read audio duration from '{audio_path.name}': {self._subprocess_message(exc)}"
            ) from exc
        except ValueError as exc:
            raise AudioProcessingError(f"Invalid duration value from '{audio_path.name}'") from exc

    def get_file_metadata(self, file_path: Path) -> FileMetadata:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            raw_metadata = json.loads(result.stdout)
        except subprocess.CalledProcessError as exc:
            raise AudioProcessingError(
                f"Failed to read audio metadata from '{file_path.name}': {self._subprocess_message(exc)}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise AudioProcessingError(f"Invalid metadata format from '{file_path.name}'") from exc
        format_info = raw_metadata.get("format", {})
        stream_info = raw_metadata.get("streams", [{}])[0] if raw_metadata.get("streams") else {}
        if not format_info:
            raise AudioProcessingError(f"No format information found in '{file_path.name}'")

        try:
            return FileMetadata(
                filename=format_info.get("filename", ""),
                path=str(file_path),
                size_bytes=int(format_info.get("size", 0)),
                duration=float(format_info.get("duration", 0)),
                format=format_info.get("format_name", ""),
                bit_rate=int(format_info.get("bit_rate", 0)),
                codec=stream_info.get("codec_name", ""),
                sample_rate=int(stream_info.get("sample_rate", 0) or 0),
                channels=int(stream_info.get("channels", 0) or 0),
            )
        except (TypeError, ValueError) as exc:
            raise AudioProcessingError(f"Invalid metadata values from '{file_path.name}'") from exc


class AudioChunker:
    """Split long audio files into overlapping wav chunks."""

    def __init__(self, inspector: AudioInspector):
        self.inspector = inspector

    def chunk_audio(
        self,
        audio_path: Path,
        chunk_dir: Path,
        duration_sec: int,
        overlap_sec: int,
    ) -> List[Path]:
        if duration_sec <= overlap_sec:
            raise ValidationError("chunk duration must be greater than overlap")

        total_duration = self.inspector.get_duration(audio_path)
        chunk_paths = []
        start_times = []
        current_start = 0.0
        while current_start < total_duration:
            start_times.append(current_start)
            current_start += duration_sec - overlap_sec

        chunk_dir.mkdir(parents=True, exist_ok=True)
        for index, start_time in enumerate(start_times):
            end_time = min(start_time + duration_sec, total_duration)
            chunk_path = chunk_dir / f"chunk_{index:03d}.wav"
            self._create_chunk(audio_path, chunk_path, start_time, end_time)
            chunk_paths.append(chunk_path)
        return chunk_paths

    def _create_chunk(
        self,
        audio_path: Path,
        chunk_path: Path,
        start_time: float,
        end_time: float,
    ) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(chunk_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            output = exc.stderr or exc.stdout or "Unknown error"
            raise AudioProcessingError(
                f"Failed to create chunk for '{audio_path.name}': {output.strip() or 'Unknown error'}"
            ) from exc


def merge_transcripts(chunk_results: List[TranscriptResult], overlap_sec: int) -> TranscriptResult:
    """Merge chunk results into one transcript."""
    if not chunk_results:
        return TranscriptResult(text="", segments=[], provider="merged")
    if len(chunk_results) == 1:
        return chunk_results[0]

    merged_text = []
    merged_segments: List[TranscriptSegment] = []
    segment_offset = 0.0

    for index, result in enumerate(chunk_results):
        if index == 0:
            for segment in result.segments:
                merged_segments.append(segment)
                if segment.text:
                    merged_text.append(segment.text)
        else:
            previous_texts = {segment.text.strip().lower() for segment in merged_segments[-10:]}
            for segment in result.segments:
                if segment.start < overlap_sec and segment.text.strip().lower() in previous_texts:
                    continue
                merged_segments.append(
                    TranscriptSegment(
                        id=segment.id,
                        start=segment.start + segment_offset,
                        end=segment.end + segment_offset,
                        text=segment.text,
                        provider_data=segment.provider_data,
                    )
                )
                if segment.text:
                    merged_text.append(segment.text)

        if result.segments:
            segment_offset = max(result.segments[-1].end - overlap_sec, 0.0)

    return TranscriptResult(
        text=" ".join(part for part in merged_text if part).strip(),
        segments=merged_segments,
        provider="merged",
        model="multi-provider",
    )
