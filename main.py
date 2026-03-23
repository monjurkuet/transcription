#!/usr/bin/env python3
"""
Audio Transcription Script with Chunking
Transcribes all audio files in a directory using Groq Whisper API.
Handles large files by splitting into chunks.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from groq_key_manager import RoundRobinKeyManager

# Supported audio file extensions
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".flac",
    ".m4a",
    ".ogg",
    ".webm",
    ".mpeg",
    ".mpga",
}

# Groq API settings
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3"
RESPONSE_FORMAT = "verbose_json"
TIMESTAMP_GRANULARITIES = ["segment"]

# Chunk settings
CHUNK_DURATION_SEC = 600  # 10 minutes
CHUNK_OVERLAP_SEC = 5  # 5 seconds overlap
MAX_FILE_SIZE_MB = 25  # Groq free tier limit (use 100 for dev tier)


def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger("transcription")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """Get comprehensive file metadata using ffprobe."""
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw_metadata = json.loads(result.stdout)

    # Extract relevant information
    format_info = raw_metadata.get("format", {})
    stream_info = raw_metadata.get("streams", [{}])[0] if raw_metadata.get("streams") else {}

    return {
        "filename": format_info.get("filename", ""),
        "path": str(file_path),
        "size_bytes": int(format_info.get("size", 0)),
        "duration": float(format_info.get("duration", 0)),
        "format": format_info.get("format_name", ""),
        "bit_rate": int(format_info.get("bit_rate", 0)),
        "codec": stream_info.get("codec_name", ""),
        "sample_rate": int(stream_info.get("sample_rate", 0)),
        "channels": stream_info.get("channels", 0),
    }


def chunk_audio(
    audio_path: Path,
    chunk_dir: Path,
    duration_sec: int = CHUNK_DURATION_SEC,
    overlap_sec: int = CHUNK_OVERLAP_SEC,
) -> List[Path]:
    """
    Split audio into overlapping chunks using ffmpeg.
    Returns list of chunk file paths.
    """
    # Get total duration
    total_duration = get_audio_duration(audio_path)
    chunk_paths = []

    # Calculate chunk boundaries
    start_times = []
    current_start = 0.0
    while current_start < total_duration:
        start_times.append(current_start)
        current_start += duration_sec - overlap_sec

    # Create chunks
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for i, start_time in enumerate(start_times):
        # Calculate end time (min of chunk end or total duration)
        end_time = min(start_time + duration_sec, total_duration)

        # Output filename
        chunk_path = chunk_dir / f"chunk_{i:03d}.wav"

        # Extract chunk using ffmpeg
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            str(audio_path),
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-ar",
            "16000",  # Resample to 16kHz (optimal for Whisper)
            "-ac",
            "1",  # Mono
            "-acodec",
            "pcm_s16le",  # WAV codec
            str(chunk_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        chunk_paths.append(chunk_path)

    return chunk_paths


def transcribe_chunk(
    chunk_path: Path, key_manager: RoundRobinKeyManager, logger: logging.Logger, max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """Transcribe a single audio chunk using Groq Whisper API with round-robin keys."""
    url = GROQ_API_URL

    for attempt in range(max_retries):
        api_key = key_manager.get_key()
        if not api_key:
            logger.error("All API keys are exhausted")
            return None
        
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            with open(chunk_path, "rb") as f:
                files = {"file": (chunk_path.name, f, "audio/wav")}
                data = {
                    "model": MODEL,
                    "response_format": RESPONSE_FORMAT,
                    "timestamp_granularities[]": TIMESTAMP_GRANULARITIES,
                }

                response = requests.post(
                    url, files=files, data=data, headers=headers, timeout=300
                )

                # Report rate limit headers back to key manager
                key_manager.report_rate_limit_response(api_key, dict(response.headers))

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Mark this key as rate-limited and try next
                    retry_after = None
                    if "retry-after" in response.headers:
                        try:
                            retry_after = int(response.headers["retry-after"])
                        except ValueError:
                            pass
                    key_manager.on_rate_limit(api_key, retry_after)
                    
                    wait_time = (2**attempt) * 5
                    logger.warning(
                        f"Rate limited on key {key_manager._key_states[api_key].masked_key}, "
                        f"waiting {wait_time}s before retry with next key..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    continue

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Request error (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            continue

    return None


def merge_transcriptions(
    chunk_results: List[Dict[str, Any]], overlap_sec: int = CHUNK_OVERLAP_SEC
) -> Dict[str, Any]:
    """
    Merge transcriptions from multiple chunks with adjusted timestamps.
    Removes duplicate segments at chunk boundaries.
    """
    if not chunk_results:
        return {"text": "", "segments": []}

    # If only one chunk, return as-is
    if len(chunk_results) == 1:
        return chunk_results[0]

    merged_text = ""
    merged_segments = []
    segment_offset = 0.0

    for chunk_idx, result in enumerate(chunk_results):
        segments = result.get("segments", [])

        if chunk_idx == 0:
            # First chunk: keep all segments
            for seg in segments:
                merged_segments.append(seg)
                merged_text += seg.get("text", "") + " "
        else:
            # Subsequent chunks: adjust timestamps and filter duplicates
            overlap_samples = []  # Store text samples from overlap region

            # Get samples from overlap region in this chunk
            for seg in segments:
                if seg.get("start", 0) < overlap_sec:
                    overlap_samples.append(seg.get("text", "").strip().lower())

            # Add non-duplicate segments
            prev_texts = set()
            for seg in segments:
                seg_start = seg.get("start", 0)
                seg_text = seg.get("text", "").strip()

                # Skip segments in overlap region that appear to be duplicates
                if seg_start < overlap_sec and seg_text.lower() in prev_texts:
                    continue

                # Adjust timestamp
                adjusted_seg = seg.copy()
                adjusted_seg["start"] = seg_start + segment_offset
                adjusted_seg["end"] = seg.get("end", 0) + segment_offset
                merged_segments.append(adjusted_seg)

                if seg_text:
                    merged_text += seg_text + " "

                prev_texts.add(seg_text.lower())

        # Calculate offset for next chunk (excluding overlap)
        if chunk_results[chunk_idx].get("segments"):
            last_seg = chunk_results[chunk_idx]["segments"][-1]
            segment_offset = last_seg.get("end", 0) - overlap_sec

    return {"text": merged_text.strip(), "segments": merged_segments}


def strip_metadata_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove tokens and text field from transcription result."""
    cleaned = result.copy()
    # Remove tokens from each segment
    if "segments" in cleaned:
        cleaned["segments"] = [
            {k: v for k, v in seg.items() if k != "tokens"}
            for seg in cleaned["segments"]
        ]
    # Remove the text field (redundant - full text is concatenation of segment texts)
    cleaned.pop("text", None)
    return cleaned


def transcribe_audio_standard(
    audio_path: Path, key_manager: RoundRobinKeyManager, logger: logging.Logger, max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """Transcribe audio file using Groq Whisper API with round-robin keys."""
    url = GROQ_API_URL

    for attempt in range(max_retries):
        api_key = key_manager.get_key()
        if not api_key:
            logger.error("All API keys are exhausted")
            return None
        
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, f"audio/{audio_path.suffix[1:]}")}
                data = {
                    "model": MODEL,
                    "response_format": RESPONSE_FORMAT,
                    "timestamp_granularities[]": TIMESTAMP_GRANULARITIES,
                }

                response = requests.post(
                    url, files=files, data=data, headers=headers, timeout=300
                )

                # Report rate limit headers back to key manager
                key_manager.report_rate_limit_response(api_key, dict(response.headers))

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Mark this key as rate-limited and try next
                    retry_after = None
                    if "retry-after" in response.headers:
                        try:
                            retry_after = int(response.headers["retry-after"])
                        except ValueError:
                            pass
                    key_manager.on_rate_limit(api_key, retry_after)
                    
                    wait_time = (2**attempt) * 5
                    logger.warning(
                        f"Rate limited on key {key_manager._key_states[api_key].masked_key}, "
                        f"waiting {wait_time}s before retry with next key..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    continue

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Request error (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            continue

    return None


def scan_audio_files(
    source_dir: Path, logger: logging.Logger
) -> List[Tuple[Path, Path]]:
    """Recursively scan for audio files and return list of (audio_path, relative_path)."""
    audio_files = []

    for ext in SUPPORTED_EXTENSIONS:
        for audio_path in source_dir.rglob(f"*{ext}"):
            # Skip non-audio files
            if audio_path.is_file() and audio_path.stat().st_size > 0:
                rel_path = audio_path.relative_to(source_dir)
                audio_files.append((audio_path, rel_path))

    audio_files.sort(key=lambda x: x[1])
    logger.info(f"Found {len(audio_files)} audio files in {source_dir}")
    return audio_files


def create_output_structure(
    source_dir: Path,
    output_dir: Path,
    audio_files: List[Tuple[Path, Path]],
    logger: logging.Logger,
) -> None:
    """Create output directory structure mirroring source."""
    output_dir.mkdir(parents=True, exist_ok=True)

    created_dirs = set()
    for _, rel_path in audio_files:
        out_subdir = output_dir / rel_path.parent
        if out_subdir not in created_dirs:
            out_subdir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(out_subdir)

    logger.info(f"Created output directory structure at {output_dir}")


def get_output_paths(audio_rel_path: Path, output_dir: Path) -> Path:
    """Get output .json path for an audio file."""
    return output_dir / audio_rel_path.with_suffix(".json")


def process_file(
    audio_path: Path,
    rel_path: Path,
    output_dir: Path,
    key_manager: RoundRobinKeyManager,
    logger: logging.Logger,
) -> bool:
    """Process a single audio file. Returns True if successful."""
    json_path = get_output_paths(rel_path, output_dir)

    # Check if already transcribed
    if json_path.exists():
        logger.info(f"Skipping {rel_path} (already transcribed)")
        return True

    try:
        logger.info(f"Transcribing: {rel_path}")

        # Get file size
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.1f} MB")

        # Check if we need chunking
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.info(f"File exceeds {MAX_FILE_SIZE_MB}MB, using chunking...")

            # Create temp directory for chunks
            with tempfile.TemporaryDirectory() as chunk_dir:
                chunk_dir = Path(chunk_dir)

                # Get audio duration
                duration = get_audio_duration(audio_path)
                logger.info(f"Audio duration: {duration / 60:.1f} minutes")

                # Split into chunks
                chunk_paths = chunk_audio(audio_path, chunk_dir)
                logger.info(f"Created {len(chunk_paths)} chunks")

                # Transcribe each chunk
                chunk_results = []
                for i, chunk_path in enumerate(chunk_paths):
                    logger.info(f"Transcribing chunk {i + 1}/{len(chunk_paths)}")
                    result = transcribe_chunk(chunk_path, key_manager, logger)
                    if result:
                        chunk_results.append(result)
                    else:
                        logger.error(f"Failed to transcribe chunk {i + 1}")
                        return False

                # Merge results
                merged = merge_transcriptions(chunk_results, CHUNK_OVERLAP_SEC)

        else:
            # Standard transcription
            merged = transcribe_audio_standard(audio_path, key_manager, logger)
            if not merged:
                logger.error(f"Failed to transcribe {rel_path}")
                return False

        # Get file metadata
        file_metadata = get_file_metadata(audio_path)

        # Strip tokens and text from transcription result
        cleaned_result = strip_metadata_fields(merged)

        # Combine file metadata with transcription
        output_data = {
            "file_metadata": file_metadata,
            **cleaned_result,
        }

        # Save JSON output
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved: {json_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to transcribe {rel_path}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Groq Whisper API"
    )
    parser.add_argument(
        "source", type=Path, help="Source file or directory containing audio files"
    )
    parser.add_argument(
        "--output", type=Path, help="Output directory (default: same as source or {source}_Transcripts)"
    )
    parser.add_argument("--model", type=str, default=MODEL, help="Whisper model to use")
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=CHUNK_DURATION_SEC,
        help="Chunk duration in seconds",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP_SEC,
        help="Chunk overlap in seconds",
    )

    args = parser.parse_args()

    source = args.source.resolve()
    is_single_file = source.is_file()

    # Handle single file vs directory
    if is_single_file:
        source_dir = source.parent
        rel_path = Path(source.name)  # Just the filename, relative to source_dir
        audio_files = [(source, rel_path)]  # (absolute_path, relative_path)
    else:
        source_dir = source
        audio_files = []  # Will be populated by scan_audio_files

    # Generate output directory if not provided
    if args.output:
        output_dir = args.output.resolve()
    elif is_single_file:
        output_dir = source_dir / f"{source.stem}_Transcript"
    else:
        output_dir = source_dir.parent / f"{source_dir.name}_Transcripts"

    # Create output directory first
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging first
    log_file = output_dir / "transcription.log"
    logger = setup_logging(str(log_file))

    # Load API keys with round-robin manager
    load_dotenv()
    try:
        key_manager = RoundRobinKeyManager()
        logger.info(f"Loaded {key_manager.num_keys} API keys for round-robin")
    except ValueError as e:
        logger.error(f"Failed to load API keys: {e}")
        print(f"Error: {e}")
        print("Please set GROQ_API_KEYS (comma-separated) or GROQ_API_KEY in your .env file")
        sys.exit(1)

    if is_single_file:
        logger.info(f"Transcribing single file: {source.name}")
    else:
        logger.info(f"Source directory: {source_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(
        f"Chunk duration: {args.chunk_duration}s, Overlap: {args.chunk_overlap}s"
    )

    # Scan for audio files (only if directory)
    if not is_single_file:
        audio_files = scan_audio_files(source_dir, logger)

    if not audio_files:
        logger.warning("No audio files found!")
        sys.exit(0)

    # Create output structure
    create_output_structure(source_dir, output_dir, audio_files, logger)

    # Process files with progress bar
    success_count = 0
    fail_count = 0

    for audio_path, rel_path in tqdm(audio_files, desc="Transcribing", unit="file"):
        if process_file(audio_path, rel_path, output_dir, key_manager, logger):
            success_count += 1
        else:
            fail_count += 1

    logger.info(
        f"Transcription complete: {success_count} successful, {fail_count} failed"
    )
    print(f"\nDone: {success_count}/{len(audio_files)} files transcribed successfully")


if __name__ == "__main__":
    main()
